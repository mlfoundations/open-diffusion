"""
Copied and modified from https://github.com/mlfoundations/open_clip/blob/main/src/training/distributed.py
"""

import os

import torch


def is_global_master(config):
    return config.system.global_rank == 0


def is_global_master_from_env():
    local_rank, global_rank, world_size = world_info_from_env()
    return global_rank == 0


def is_local_master(config):
    return config.system.local_rank == 0


def is_master(config, local=False):
    return is_local_master(config) if local else is_global_master(config)


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1

    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1

    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(config):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    config.system.distributed = False

    config.system.world_size = 1
    config.system.global_rank = 0  # global rank
    config.system.local_rank = 0

    if is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            config.system.local_rank, config.system.global_rank, config.system.world_size = world_info_from_env()

            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(config.system.local_rank)
            os.environ['RANK'] = str(config.system.global_rank)
            os.environ['WORLD_SIZE'] = str(config.system.world_size)

            torch.distributed.init_process_group(
                backend=config.system.dist_backend,
                init_method=config.system.dist_url,
                world_size=config.system.world_size,
                rank=config.system.global_rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            config.system.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=config.system.dist_backend,
                init_method=config.system.dist_url)

            config.system.world_size = torch.distributed.get_world_size()
            config.system.global_rank = torch.distributed.get_rank()

        config.system.distributed = True

    if torch.cuda.is_available():
        if config.system.distributed:
            device = f'cuda:{config.system.local_rank}'
        else:
            device = 'cuda:0'

        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    config.system.device = device
    device = torch.device(device)

    return device


def compute_effective_batch_size(worker_batch_size):
    if is_using_distributed():
        _, _, world_size = world_info_from_env()
        return world_size * worker_batch_size
    else:
        return worker_batch_size
