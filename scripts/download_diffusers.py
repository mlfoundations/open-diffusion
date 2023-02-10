from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from utils.logging import Path


import torch
import argparse


def main():
    args = get_args()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, scheduler=scheduler).to('cpu')

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    pipe.save_pretrained(args.output)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, help="Pipeline output directory")
    parser.add_argument("-m", "--model-id", type=Path, help="Model ID (huggingface hub address)")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()



