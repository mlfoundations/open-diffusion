from omegaconf import OmegaConf
from utils.logging import Path

import torch
import os


def get_optimizer(
    model, learning_rate, beta1, beta2, weight_decay, epsilon 
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=epsilon,
    )


def get_config():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf 


def get_experiment_folder(config):
    log_dir = Path(config.experiment.log_dir)
    folder = log_dir / config.experiment.name
    os.makedirs(folder, exist_ok=True)

    return folder

