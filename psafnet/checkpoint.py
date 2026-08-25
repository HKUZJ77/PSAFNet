"""Checkpoint helpers shared by the public training and test scripts."""

from collections import OrderedDict
from pathlib import Path

import torch


def _load_file(path, map_location):
    try:
        return torch.load(
            str(path),
            map_location=map_location,
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        # PyTorch versions before weights_only/mmap support.
        return torch.load(str(path), map_location=map_location)


def _model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a model state dictionary.")

    state = OrderedDict()
    for name, value in checkpoint.items():
        clean_name = name[7:] if name.startswith("module.") else name
        state[clean_name] = value
    return state


def load_psafnet_checkpoint(model, checkpoint_path, map_location="cpu", strict=True):
    """Load either a training checkpoint or an inference-only state dictionary."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    checkpoint = _load_file(checkpoint_path, map_location)
    state = _model_state(checkpoint)
    model.load_state_dict(state, strict=strict)

    metadata = {}
    if isinstance(checkpoint, dict):
        for key in ("epoch", "best_loss", "dataset", "fov", "coordinate"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    return metadata
