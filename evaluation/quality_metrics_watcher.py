from utils.logging import Path

import argparse
import clip
import wandb

from evaluation.quality_metrics import get_step_file_index
from evaluation.quality_metrics import extract_files_from_tarpath
from evaluation.quality_metrics import image_from_file, compute_clip_score

import pandas as pd
import tqdm


def main():
    config, args = get_args()

    # Loading the relevant CLIP model
    model, preprocess = clip.load("ViT-L/14")
    model = model.to(0)

    wandb.login(key=config.wandb.api_key)
    wandb.init(
        project=config.experiment.project,
        name=config.experiment.name,
        id=config.wandb.run_id,
        resume="allow",
        dir=args.example_dir,
    )

    example_dir = Path(args.example_dir)

    # find all step tarfiles in example_dir and sort them
    indices_and_tarfiles = sorted(
        [(get_step_file_index(p), p) for p in example_dir.glob("*.tar")]
    )

    reference = pd.read_csv(args.ref_path)

    for step, tar_path in tqdm.tqdm(indices_and_tarfiles):
        tar_out = extract_files_from_tarpath(
            tar_path,
            preprocess=lambda f: preprocess(image_from_file(f)),
            filter_fn=lambda member: member.name.endswith("jpg"),
        )

        filenames, images = zip(*tar_out)

        prompts = [
            reference[reference.uid == str(Path(f).stem)].caption.values[0]
            for f in filenames
        ]

        clip_score = compute_clip_score(model, torch.stack(images), prompts)

        wandb.log({"clip_score_": clip_score.item(), "epoch": step})

    # Images now contains all images in the tar file
    globals().update(locals())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-path", type=Path)
    parser.add_argument("--prompts", type=Path, default="data/prompts/uid_caption.csv")

    return parser.parse_args()

