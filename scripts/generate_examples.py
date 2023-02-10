"""
Simple script to generate examples with a pretrained pipeline

Can be used with srun or torchrun to distribute across many nodes
"""

# TODO: add resuming
import sys

from utils.logging import tar_and_remove_dir

sys.path.append(".")

from utils.distributed import world_info_from_env, init_distributed_device
from utils.data_utils import batchify

from diffusers import StableDiffusionPipeline

from pathlib import Path
import os

from torch import autocast
import torch.distributed as dist
import torch

from argparse import ArgumentParser
from omegaconf import OmegaConf
import pandas as pd
import math
import subprocess
import shutil


def main():
    args = get_args()
    config = OmegaConf.from_dotlist(["system.dummy=None"])

    # To initialize process group
    # device = init_distributed_device(config)

    prompt_db = pd.read_csv(args.prompt_file)
    local_rank, global_rank, world_size = world_info_from_env()
    device = torch.device(f"cuda:{local_rank}")


    # select partition denoted by world_size and global_rank
    chunk_size = math.ceil(len(prompt_db) / world_size)
    start_idx = chunk_size * global_rank
    prompt_db = prompt_db.iloc[start_idx : (start_idx + chunk_size), :]
    pipeline = get_diffuser(args.pipeline, device=device)

    out_dir = args.out_dir / f"shard_{global_rank}"

    if not concatenate_to_path(out_dir, ".tar").exists():
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.scratch_dir / out_dir.stem, exist_ok=True)

        with torch.no_grad():
            with autocast('cuda', dtype=torch.float16):
                for batch_db in batchify(prompt_db, batch_size=args.batch_size):
                    images = generate_image_from_prompt(
                        prompts=batch_db.caption.to_list(),
                        pipe=pipeline,
                        guidance_scale=args.guidance_scale,
                    )

                    for image, uid in zip(images, batch_db.uid):
                        image.save(args.scratch_dir / out_dir.stem / f"{uid}.jpg")
        
        if args.tar:
            tar_and_remove_dir(out_dir=args.scratch_dir / out_dir.stem, target_dir=out_dir.parent)
        else:
            shutil.move(args.scratch_dir / out_dir.stem, out_dir)

    # Merge shards
    # dist.barrier()

    # if global_rank == 0:
    #     # Merge all shards
    #     shard_list = out_dir.parent.glob("shard_*")
    #     subprocess.Popen()


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to prompt json of the format prompt -> foldername",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs"), help="Where to put output files"
    )
    parser.add_argument("--scratch-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per device batch size"
    )
    parser.add_argument(
        "--guidance-scale",
        default=7.5,
        type=float,
        help="Guidance weight to use for generation",
    )
    parser.add_argument("--pipeline", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--tar", default=None, type=Path)

    args = parser.parse_args()

    return args


def get_diffuser(pt_path, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        pt_path,
        use_auth_token=True,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)

    return pipe


def tar_folder(folder_path, out_path):
    os.system(f"tar czvfP {out_path} -C {folder_path.parent} {folder_path.name}")

    return out_path


def post_process_image(pth_image):
    image = (pth_image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    return image


def get_rng(seed, device="cuda:0"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    return gen


def generate_image_from_prompt(prompts, pipe, seed=0, guidance_scale=7.5):
    rng = get_rng(seed=seed)
    outs = pipe(
        prompts,
        guidance_scale=guidance_scale,
        generator=rng,
    )

    return outs.images


def concatenate_to_path(path: Path, suffix):
    return path.parent / f"{path.stem}{suffix}"


if __name__ == "__main__":
    main()