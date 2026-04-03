"""Shared runtime validation helpers for pipeline entry scripts."""

import importlib.util
import os


def _is_missing(value):
    """Return True when a config value should be treated as unset."""
    return value is None or (isinstance(value, str) and not value.strip())


def require_env_value(name, value, extra_help=None):
    """Raise a clear error if an expected environment-derived value is missing."""
    if _is_missing(value):
        message = f"Missing required configuration value: {name}."
        if extra_help:
            message = f"{message} {extra_help}"
        raise RuntimeError(message)
    return value


def require_existing_file(path, description):
    """Raise a clear error if an expected file is missing."""
    require_env_value(description, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_existing_dir(path, description):
    """Raise a clear error if an expected directory is missing."""
    require_env_value(description, path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_cuda_gpus(script_name, require_xformers=False):
    """Validate CUDA availability and return the visible GPU count."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{script_name} requires a CUDA-capable GPU and a CUDA-enabled PyTorch install."
        )

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError(f"{script_name} requires at least one visible CUDA GPU.")

    if require_xformers and importlib.util.find_spec("xformers") is None:
        raise RuntimeError(
            f"{script_name} requires the optional 'xformers' package because it calls "
            "enable_xformers_memory_efficient_attention()."
        )

    return world_size
