from utils.logging import Path
from omegaconf import OmegaConf
import torch
import json
import importlib


def maybe_load_model(config, subtype, default_model_factory=None):
    model_factory = load_target(config.model.get(f"{subtype}.target"), default=default_model_factory)
    if model_id := config.model.get("pretrained"):
        model = model_factory.from_pretrained(model_id, subfolder=subtype)

    elif config.model.get(subtype, False) and (
        model_id := config.model.get(subtype).get("pretrained")
    ):
        model = model_factory.from_pretrained(model_id, subfolder=subtype)

    elif model_params := config.model.get(subtype).get("params"):
        model = model_factory(**OmegaConf.to_container(model_params))

    else:
        raise ValueError(f"Config missing model config for {subtype}")

    return model


def load_target(dot_path, default=None):
    if dot_path is None:
        return default

    base_module = importlib.import_module(dot_path.split(".")[0])

    return getattr(base_module, ".".join(dot_path.split(".")[1:]), default)


@torch.no_grad()
def grad_norm_sum(model):
    return sum(p.grad.norm() for p in model.parameters())
