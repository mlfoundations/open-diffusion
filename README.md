# Simple at-scale training of Stable Diffusion

- [Simple at-scale training of Stable Diffusion](#simple-at-scale-training-of-stable-diffusion)
  - [Purpose](#purpose)
  - [Usage](#usage)
    - [Requirements](#requirements)
    - [Configuration](#configuration)
    - [Running a large-scale experiment](#running-a-large-scale-experiment)
    - [Local Fine-tuning example](#local-fine-tuning-example)
    - [Inference](#inference)
    - [Examples](#examples)
    - [Performance and System Requirements](#performance-and-system-requirements)
  - [Recommendations for Large-Scale Training](#recommendations-for-large-scale-training)
  - [Future features](#future-features)
  - [Contributors](#contributors)
  - [Acknowledgments](#acknowledgments)

## Purpose

Our goal for this repo is two-fold:

1. Provide a transparent, simple implementation of which supports large-scale stable diffusion training for research purposes
2. Integration with existing diffusion APIs

We accomplish this through minimal scaffolding code and utilization of the `diffusers` (github.com/huggingface/diffusers) repo. In the future, we will move simple elements of the diffusers repo (e.g. the noise scheduler) into this one for better clarity and easier extensibility. We also support the WebDataset format (github.com/webdataset/webdataset) to allow for training on large-scale datasets.


## Usage

### Requirements

First install the requirements under Python 3.9 in a virtualenv:

```
pip install -r requirements.txt
```

You might need to install `xformers` directly from their GitHub repo (github.com/facebookresearch/xformers). Further, we currently support UNet from-scratch training only. The rest of the components must be pretrained. You can download a compatible model using the following command:

```
python scripts/download_diffusers.py -o out_pipeline_path -m model_id
```

You can get various model_ids from the runway-ml/stabilityai/CompVis organizations on huggingface. A useful one is "stabilityai/stable-diffusion-2-base".

Finally, you will need to set up your dataset in the WebDataset format with image (jpg, png, webp) and text (txt) fields corresponding to images and captions respectively. You can also include a json field if you want to apply metadata filters (described below). See the `img2dataset` project at https://github.com/rom1504/img2dataset for details on how to do this for large-scale datasets.


### Configuration

We use OmegaConf to handle commandline configuration. See `configs/template_config.yaml` for a template version. This is reproduced here for explanation. 

```
wandb:
    api_key: ???
    # entity: ??? # necessary for a project group

system:
    gradient_accumulation: 1
    batch_size: 32
    workers: 6
    dist_backend: ${distributed.dist_backend}
    dist_url: ${distributed.dist_url}

distributed:
    dist_backend: 'nccl'
    dist_url: 'env://'

experiment:
    log_dir: ???
    name: ???
    project: ??? 
    num_examples_to_see: ???

    save_only_most_recent: True
    save_every: 2000
    requeue: True

optimizer:
    name: adamw
    params:
        learning_rate: 0.0001
        beta1: 0.9
        beta2: 0.98 
        weight_decay: 0.01
        epsilon: 0.00000001

model:
    vae:
        pretrained: ???

    text_encoder:
        pretrained: ???

    tokenizer:
        pretrained: ???
    
    scheduler:
        pretrained: ???
    
    unet:
        target: UNet2DConditionModel
        params:
            act_fn: "silu"
            attention_head_dim: 8
            block_out_channels: [320, 640, 1280, 1280]
            center_input_sample: False
            cross_attention_dim: 768
            down_block_types: ["CrossAttnDownBlock2D","CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]
            downsample_padding: 1
            flip_sin_to_cos: true
            freq_shift: 0
            in_channels: 4
            layers_per_block: 2
            mid_block_scale_factor: 1
            norm_eps: 1e-05
            norm_num_groups: 32
            out_channels: 4
            sample_size: 32
            up_block_types: [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ]
        
    use_ema: False
    mixed_precision: bf16
    gradient_checkpointing: True
    xformers: True

dataset:
    type: WebDataset
    params: 
        path: ???
        batch_size: ${system.batch_size}
        workers: ${system.workers}
        num_examples_to_see: ${experiment.num_examples_to_see}
        resolution: 512

        filters: # optional metadata filters
            # original_width: ">=512"
            # original_height: ">=512"
            pwatermark: "<=0.5"


lr_scheduler:
    scheduler: "ConstantWithWarmup"
    params:
        learning_rate: ${optimizer.params.learning_rate}
        warmup_length: 500

```

Arguments with `???` are required. 

- `wandb.api_key` -> This is the per user API key you get from `wandb.ai` when you sign up for an account. This repo doesn't currently support training without wandb tracking.
- `system.gradient_accumulation` -> The number of gradient accumulation steps. 
- `system.batch_size` -> The per-gpu batch size. Effective batch size per iteration is computed as `grad_accum * batch_size * num_gpus`.
- `system.workers` -> Number of workers per gpu. Should be equal to number of CPUs per task, if you are using SLURM.
- `experiment.log_dir` -> Top level log directory for the experiment.
- `experiment.project` -> Name of project (this is also passed to wandb as the project name)
- `experiment.name` -> Name of this particular experiment. Logs will go in the folder `log_dir/project/name`. This name is also passed to wandb.
- `experiment.save_every` -> Saves every `save_every * effective_batch_size` images seen.
- `model.pretrained` -> This is the initial model to start with. Can be a huggingface address or a local folder name. To train from scratch, see the example in `configs/laion/from-scratch.yaml`. 
- `dataset.params.path` -> Path to webdataset shards, which is a url in brace notation. See `github.com/webdataset/webdataset` for more details about the webdataset format.
- `dataset.params.filters` -> Optionally provide filters based on WebDataset metadata. WebDataset entries must have a "json" field to work. Entries without metadata filter names provided here are discarded.
- `model.*` 
  - There are 4 key model parameters to declare:
    1. VAE
    2. UNet 
    3. Tokenizer
    4. Scheduler 

    You can declare each of these with the `pretrained` argument, which should go to the root of a pretrained diffusers pipeline, or with the `params` argument, which corresponds to arguments passed directly to the target constructor. The pattern above is the intended one, where only the UNet is fine-tuned. This repo currently supports from-scratch UNet training, so we declare all other parts pretrained (although only the VAE can really be considered pretrained). You can get the other pretrained components from `github.com/huggingface/diffusers`. If you would like to fine-tune, you don't have to declare any subcomponents and instead just declare `model.pretrained` directly.
  - `xformers` -> Whether or not to use `xformers` memory efficient attention. To use this you need to install `xformers` (see `github.com/facebookresearch/xformers`). It's very recommended, usually cuts GPU memory consumption in half. 
  - `use_ema` -> Whether or not to use Exponential Moving Average training. This requires more VRAM but improves training stability, and is useful for very long training runs on large datasets. 
  - `ema.path` -> Path to a pretrained EMA model. Usually only relevant for multi-stage training or fine-tuning on a large dataset.


### Running a large-scale experiment

We use SLURM to support multi-node training. Below is a general template for an SBATCH script. This will likely need to be modified to work on your individual cluster. The flags in `configs/batch_files/multinode-template.sbatch` are necessary for running on an AWS SLURM cluster.

```
#!/bin/bash
#SBATCH --job-name=Job_Name
#SBATCH --partition=Partition_Name
#SBATCH --account=laion
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --time=96:00:00
#SBATCH --chdir=/path/to/sd-training
#SBATCH --output=/stdout/path
#SBATCH --error=/stderr/path
#SBATCH --requeue
#SBATCH --exclusive


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802 # This is an arbitrary choice, change if this port is taken

srun --cpu_bind=v --accel-bind=gn \
    python -u train.py \
    config=path/to/yaml/config
```

`--ntasks-per-node` should equal the number of GPUs per node. 

With OmegaConf, commandline overrides are done in dot-notation format. E.g. if you want to override the dataset path, you would use the command `python -u train.py config=path/to/config dataset.params.path=path/to/dataset`. If you're using the `WebDataset` type, this path also accepts pipes to AWS sharded datasets. 


### Local Fine-tuning example

This requires at least one GPU with >=24 GB of VRAM (e.g. a 3090). 

Below is an example config:

```
wandb:
    api_key: ???

system:
    gradient_accumulation: 1
    batch_size: 2
    workers: 6
    dist_backend: ${distributed.dist_backend}
    dist_url: ${distributed.dist_url}

distributed:
    dist_backend: 'nccl'
    dist_url: 'env://'

experiment:
    log_dir: logs
    name: "finetuning"
    project: "diffusion"

    num_examples_to_see: ???

    eval_caption_file: ???
    num_eval_images: 1000
    save_every: 1000
    requeue: True

optimizer:
    name: adamw
    params:
        learning_rate: 0.0001
        beta1: 0.9
        beta2: 0.98 # changed from initial sd value for training stability
        weight_decay: 0.01
        epsilon: 0.00000001

model:
    pretrained: ???
    use_ema: False
    mixed_precision: bf16
    gradient_checkpointing: True
    xformers: True

dataset:
    type: WebDataset
    params: 
        path: ???
        batch_size: ${system.batch_size}
        workers: ${system.workers}
        num_examples_to_see: ${experiment.num_examples_to_see}
        resolution: 512

lr_scheduler:
    scheduler: "ConstantWithWarmup"
    params:
        learning_rate: ${optimizer.params.learning_rate}
        warmup_length: 500
```

Fill in every field with a `???`, explanations for these variables can be found in the previous section and save this config. Next, you can run this experiment using `torchrun` (a torch specific version of `mpirun`), as below:

```bash
torchrun --nproc_per_node "# of gpus" -m train config=path/to/config
```

We currently support datasets in the `webdataset` or `parquet` formats. These datasets must have text (`txt`) and image fields (`jpg`, `png`, `webp`). We also support classification datasets with a text field provided by (`cls`). In this context you would also need to provide the variable `dataset.params.idx_to_prompt`, which is a json mapping from `cls` to prompt. See `scripts/metadata/imagenet_idx_to_prompt.json` for examples of this. In the future we will support the CSV dataset format as well. 

### Inference

Given our integration with the diffusers library, you can use their API for generation with any saved pipeline. Your latest model will be saved at `experiment.log_dir / experiment.name / current_pipeline`. 


Example inference code snippet:

```python
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

pt_path = "path/to/pretrained/pipeline"

scheduler = EulerDiscreteScheduler.from_pretrained(pt_path, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(pt_path, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("output.png")
```


### Examples


Below are some random (non-cherry picked) examples from training from scratch on ImageNet. See `configs/templates/imagenet.yaml` for the exact setup of this experiment.

![ImageNet Examples](images/imagenet-examples.png)

More examples from a large-scale training run to come. See `configs/templates` for stage 1 and stage 2 of training SD v2 base. We also include a script `scripts/generate_examples.py` to help for large-scale generation of images from a prompt list. 


### Performance and System Requirements

Mixed precision currently only supports `bfloat16`, which requires 30xx NVIDIA GPUs or A40s/100s. With all optimizations enabled (gradient checkpointing, mixed precision, and xformers) and EMA, we achieve the following throughputs (including distributed training overhead) on 80GB A100s through an AWS SLURM cluster:

- Resolution 256x256: ~38 images/sec/GPU with batch size 256 per GPU
- Resolution 512x512: ~12 images/sec/GPU with batch size with batch size 64 per GPU 

Note that aggressive WebDataset filtering, throughput decreases due to the data loading bottleneck. Also note that GPU memory (~60%) is not saturated in either of these benchmarks.

We use the following architecture parameters for this performance benchmark:

```yaml
  unet:
      target: diffusers.UNet2DConditionModel
      params:
          act_fn: "silu"
          attention_head_dim: 8
          block_out_channels: [320, 640, 1280, 1280]
          center_input_sample: False
          cross_attention_dim: 768
          down_block_types: ["CrossAttnDownBlock2D","CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]
          downsample_padding: 1
          flip_sin_to_cos: true
          freq_shift: 0
          in_channels: 4
          layers_per_block: 2
          mid_block_scale_factor: 1
          norm_eps: 1e-05
          norm_num_groups: 32
          out_channels: 4
          sample_size: 32
          up_block_types: [
              "UpBlock2D",
              "CrossAttnUpBlock2D",
              "CrossAttnUpBlock2D",
              "CrossAttnUpBlock2D"
          ]

```


## Recommendations for Large-Scale Training

- **Training Resolution:** As of now, the pretrained VAE used with Stable Diffusion does not perform as well at 256x256 resolution as 512x512. In particular, faces and intricate patterns become distorted upon compression. This makes SD training worse/lower quality. If you are going to train at 256x256, you should include some fine-tuning at 512x512 for better quality generations.
- **Data Resolution:** The native data resolution is also key for producing high quality outputs. You should fine-tune on high-res or ultra high-res data at some point (at least 512x512 native resolution).
- **Training duration:** In general, the longer the better. For datasets on the scale of hundreds of millions of images you will want ot tr
- **Batch size:** We recommend training with a large global batch size (e.g. 1024 or 2048).
- **Model size:** Usually, the larger the better. We currently don't support FSDP (this is on the roadmap)
- **Filtering:** If you provide metadata in a `json` field along with your webdataset, you can filter as described above. If you provide too many restrictive filters data loading will become quite slow, in which case you should re-shard your dataset by the most restrictive filters (e.g. construct subset of LAION-2b with only original width/height >=512 images). Besides resolution filtering, watermark filtering is pretty important to achieving good outputs. Watermark scores are provided with the LAION-2b metadata.


## Future features

In this order:

- [ ] CSV dataset support
- [ ] Pre-trained ImageNet/large-scale checkpoints
- [ ] Support for alternative text encoder models (already supported, just not documented)
- [ ] S3 path support (partially supported)
- [ ] Metrics for pre-trained models
- [ ] AutoEncoder training
- [ ] Efficient FSDP
- [ ] Support for alternative text encoders
- [ ] Asynchronous FID/CLIPScore validation


## Contributors

The current version of the repo was authored by Vivek Ramanujan. Thank you to Robin Rombach, Rom Beaumont, and Jon Hayase for feedback and assistance. If you would like to contribute, open a PR/issue!


## Acknowledgments

Thank you to Stability for providing the resources to test this code. A further thanks to the excellent open-source repos this one depends on, including `diffusers`, `xformers`, `webdataset`, and of course `pytorch`.