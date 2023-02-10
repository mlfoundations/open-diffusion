import os
import pathlib
import cloudpathlib
import time
from omegaconf import DictConfig, ListConfig, OmegaConf

from omegaconf import OmegaConf
from typing import List, Tuple, Any

import shutil


def flatten_dict(d):
    flat_dict = {}
    queue = [(d, k, []) for k in d.keys()]

    while queue:
        parent_dict, node, path = queue.pop()
        child = parent_dict[node]
        new_path = path + [node]

        if isinstance(child, dict):
            queue.extend([(child, k, new_path) for k in child.keys()])
        else:
            flat_dict[".".join(new_path)] = child

    return flat_dict


def dict_from_flatten(flatlist: List[Tuple[str, Any]]) -> DictConfig:
    return OmegaConf.from_dotlist([f"{k}={v}" for k, v in flatlist])


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret

def override_arguments(command_line):
    eval()


def tar_and_remove_dir(out_dir: pathlib.Path, target_dir=None, remove=True):
    base_dir = out_dir.absolute().parent

    shutil.make_archive(
        base_name=base_dir / out_dir.name,
        format="tar",
        root_dir=base_dir,
        base_dir=out_dir.name,
    )

    if remove:
        shutil.rmtree(out_dir.absolute())

    if target_dir is not None:
        shutil.move(base_dir / f"{out_dir.stem}.tar", target_dir / f"{out_dir.stem}.tar")
    else:
        target_dir = base_dir

    return target_dir / f"{out_dir.stem}.tar"


def previous_experiment_path(config) -> pathlib.Path:
    return Path(config.experiment.folder) / "current_pipeline"


def Path(path) -> pathlib.Path:
    if str(path).startswith("s3://"):
        return cloudpathlib.S3Path(path)
    else:
        return pathlib.Path(path)
    
