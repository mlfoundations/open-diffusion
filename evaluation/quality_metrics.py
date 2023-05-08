import sys

import tqdm

sys.path.append(".")

from omegaconf import OmegaConf
from torchvision.transforms import transforms
from torchvision.datasets.folder import default_loader
import torchvision

import clip
import numpy as np
import argparse
from utils.logging import Path

from utils.getters import get_experiment_folder

from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

import tarfile
import io
import pandas as pd
import scipy.linalg as linalg


def main():
    base_path = Path("/home/ramanv/misc")
    step_paths = list(base_path.glob("*.tar"))
    data = []
    fid_data = []

    for tarpath in tqdm.tqdm(step_paths, ascii=True):
        step = int(tarpath.stem.split("_")[-1])
        clip_score = (
            coco_clip_score(tarpath, "data/prompts/uid_caption.csv").cpu().item()
        )
        print(clip_score)

        fid_score = coco_fid_score(tarpath, "/usr/data/mscoco/mscoco.tar")
        print(fid_score)

        data.append((step, clip_score))
        fid_data.append((step, fid_score))

    plt.figure()
    x, y = zip(*sorted(data))
    plt.plot(x, y, linewidth=3, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("CLIP Score")
    plt.tight_layout()
    plt.savefig("test.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

    plt.figure()
    x, y = zip(*sorted(fid_data))
    plt.plot(x, y, linewidth=3, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("CLIP Score")
    plt.tight_layout()
    plt.savefig("fid_test.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

    globals().update(locals())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-path", type=Path)
    parser.add_argument("--prompts", type=Path, default="data/prompts/uid_caption.csv")
    parser.add_argument("--scratch-dir", type=Path, default=Path("/dev/shm"))

    return parser.parse_args()


def coco_fid_score(
    gen_tar_path,
    ref_tar_path,
    device="cuda:0",
    model_type="ViT-L/14",
    batch_size=256,
):
    model, preprocess = clip.load(model_type, device=device)

    gen_dataset = TarImageDataset(gen_tar_path, transform=preprocess)
    gen_loader = DataLoader(
        gen_dataset, shuffle=False, batch_size=batch_size, num_workers=1
    )

    ref_dataset = TarImageDataset(ref_tar_path, transform=preprocess)
    if len(ref_dataset) > len(gen_dataset):
        n = len(gen_dataset)
        chosen = np.random.permutation(len(ref_dataset))[:n]
        ref_dataset.image_list = [ref_dataset.image_list[i] for i in chosen]

    ref_loader = DataLoader(
        ref_dataset, shuffle=False, batch_size=batch_size, num_workers=1
    )

    return compute_fclip_score(model, gen_loader, ref_loader)


def coco_clip_score(
    ref_tar_path, prompt_db_path, model_type="ViT-L/14", device="cuda:0", batch_size=256
):
    model, preprocess = clip.load(model_type, device=device)

    db = pd.read_csv(prompt_db_path)
    uids = db.uid.to_list()
    captions = db.caption.to_list()

    dataset = TarImageDataset(
        tar_path=ref_tar_path,
        transform=preprocess,
        ext="jpg",
        target_transform=alt_tar_path_to_prompt(uids, captions),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return compute_clip_score(model, loader)


@torch.no_grad()
def compute_fclip_score(model, loader1, loader2, device=0):
    print("=> Getting activations for loader1")
    a_1 = get_activations(model, loader1, device=device)

    print("=> Getting activations for loader2")
    a_2 = get_activations(model, loader2, device=device)

    mu1, sigma1 = compute_mu_and_sigma(a_1)
    mu2, sigma2 = compute_mu_and_sigma(a_2)

    return calculate_frechet_distance((mu1, sigma1), (mu2, sigma2))


@torch.no_grad()
def get_activations(model, loader, device=0):
    activations = []

    for batch, _ in loader:
        activation = model.encode_image(batch.to(0))
        activations.append(activation.cpu())

    return torch.cat(activations, dim=0)


def compute_mu_and_sigma(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(m1_stats, m2_stats, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1, sigma1 = m1_stats
    mu2, sigma2 = m2_stats
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@torch.no_grad()
def compute_clip_score(model, loader):
    model.eval()

    acc = 0.0
    for batch, prompts in loader:
        tokens = clip.tokenize(prompts).cuda()
        text_features = F.normalize(model.encode_text(tokens))
        image_features = F.normalize(model.encode_image(batch.to(0)))
        sim_scores = image_features @ text_features.T
        acc += sim_scores.diag().sum()

    return acc / len(loader.dataset)


def tar_path_to_prompt(uids, captions):
    uids_to_filenames = {
        uid: (i, coco_uid_to_filename(uid)) for i, uid in enumerate(uids)
    }
    filenames_to_uids = {
        filename: (i, uid) for uid, (i, filename) in uids_to_filenames.items()
    }

    def _transform(tarpath):
        i, uid = filenames_to_uids[tarpath.split("/")[-1]]

        return captions[i]

    return _transform


def alt_tar_path_to_prompt(uids, captions):
    uids_to_filenames = {
        uid: (i, coco_uid_to_filename(uid)) for i, uid in enumerate(uids)
    }

    def _transform(tarpath):
        i, filename = uids_to_filenames[Path(tarpath).stem]

        return captions[i]

    return _transform


def coco_uid_to_filename(uid):
    # uid format: 203635_mscoco_2
    # filename format: train2017/000000147076.jpg
    return f"{int(uid.split('_')[0]):012d}.jpg"


def extract_files_from_tarpath(tar_path, preprocess, filter_fn=None):
    filter_fn = filter_fn or (lambda x: True)
    tar = tarfile.open(tar_path)
    members = tar.getmembers()

    out_files = []
    for member in members:
        if filter_fn(member):
            f = tar.extractfile(member)
            out_files.append((member.name, preprocess(f)))

    return out_files


def get_step_file_index(step_file_path):
    index = int(Path(step_file_path).stem[len("step_") :])

    return index


def image_from_file(f):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(f)
    return img.convert("RGB")



if __name__ == "__main__":
    main()
