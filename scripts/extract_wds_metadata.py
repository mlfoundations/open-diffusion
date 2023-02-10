import sys

sys.path.append(".")
import webdataset as wds
from data.base import filter_keys, log_and_continue, tarfile_to_samples_nothrow

from data.policies import CenterCropSDTransform
from utils.data_utils import filter_no_caption_or_no_image
from torch.utils.data import default_collate

from utils.logging import Path
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import math
import json
import tqdm
import os

from utils.logging import tar_and_remove_dir


def main():
    url = "/path/to/wds"

    dataset, loader, num_batches = get_dataset_and_loader(
        url, train=True, num_examples_to_see=10000, num_workers=18
    )

    out_file = Path("../hr_metadata.jsonl")
    outs = []

    for batch in loader:
        outs.extend(batch["metadata"])
    
    out_file.write_text("\n".join([str(o, "utf-8") for o in outs]))


def filter_samples_by_fields(field_lists):
    # has_caption = ('txt' in sample)
    # has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)

    def filter_fn(sample):
        select = True

        for fields in field_lists:
            sub_select = False
            for field in fields:
                sub_select = sub_select or (field in sample)

            select = select and sub_select

        return select

    return filter_fn


def get_dataset_and_loader(
    url, train, num_examples_to_see, per_worker_batch_size=32, num_workers=32
):
    transform = CenterCropSDTransform(center_crop=True, size=512)

    pipeline = [wds.ResampledShards(url)]

    # TODO: Currently does not support validation sampling well
    # Don't split by worker and node since we're sampling with replacement
    # if train:
    #     pipeline.append(wds.shuffle(2000))

    pipeline.extend(
        [
            tarfile_to_samples_nothrow,
        ]
    )

    if train:
        pipeline.append(wds.shuffle(2000))

    pipeline.extend(
        [
            wds.select(
                filter_samples_by_fields(
                    [["txt"], ["png", "jpg", "jpeg", "webp"], ["json"]]
                )
            ),
            wds.rename(metadata="json"),
            wds.map(filter_keys(set(["metadata"]))),
            wds.batched(per_worker_batch_size, partial=not train, collation_fn=default_collate),
        ]
    )

    num_worker_batches = math.ceil(
        num_examples_to_see / (per_worker_batch_size * num_workers)
    )

    # Number of batches produced is _at least_ the requisite num_examples_to_see // effective_batch_size

    dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,  # Shuffling done in the webdataset
        num_workers=num_workers,
        persistent_workers=True,
    )

    return dataset, loader, num_worker_batches


if __name__ == "__main__":
    main()
