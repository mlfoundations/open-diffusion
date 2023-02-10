import sys, os

sys.path.append(".")

from torch.utils.data import DataLoader

import argparse
import torchvision
import webdataset as wds

from data.base import filter_keys, log_and_continue, tarfile_to_samples_nothrow
from utils.data_utils import filter_no_caption_or_no_image
from torch.utils.data import default_collate
from utils.logging import Path
from utils.logging import tar_and_remove_dir

import math
import json
import tqdm


def main():
    args = get_args()
    dataset, loader = get_dataset_and_loader(args.wds, batch_size=256, num_examples_to_see=args.num_examples)

    scratch_dir = Path(args.scratch_dir)

    os.makedirs(scratch_dir / args.name, exist_ok=True)
    write_dir = Path(args.write_dir)

    count = 0
    for batch in tqdm.tqdm(loader, total=args.num_examples // 256):
        for image in batch["pixel_values"]:
            torchvision.transforms.ToPILImage()(image).save(scratch_dir / args.name / f"{count:08d}.jpg")
            count += 1
    
    tar_and_remove_dir(out_dir=scratch_dir / args.name, target_dir=write_dir.absolute())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wds", default="pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar -")
    parser.add_argument("--scratch-dir", default="/dev/shm")
    parser.add_argument("--num-examples", type=int, default=100000)
    parser.add_argument("--write-dir", type=Path)
    parser.add_argument("--name", default="laion2b-samples")

    args = parser.parse_args()

    return args


def get_dataset_and_loader(url, batch_size, num_examples_to_see, workers=16):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
        ]
    )

    pipeline = [
        wds.ResampledShards(url),
        wds.shuffle(2000),
        tarfile_to_samples_nothrow,
        wds.shuffle(2000),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(pixel_values="jpg;png;jpeg;webp"),
        wds.map(filter_keys(set(["pixel_values"]))),
        wds.map_dict(
            pixel_values=transform,
        ),
        wds.batched(batch_size, partial=False, collation_fn=default_collate),
    ]

    num_worker_batches = math.ceil(
        num_examples_to_see / (batch_size * workers)
    )

    dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,  # Shuffling done in the webdataset
        num_workers=workers,
        persistent_workers=True,
    )

    # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size
    return dataset, loader


if __name__ == "__main__":
    main()
