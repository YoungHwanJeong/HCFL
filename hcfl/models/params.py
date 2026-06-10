# Convert trainable parameters between state dicts and flat vectors.
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def get_trainable_state(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    state = OrderedDict()
    for n, p in model.named_parameters():
        if p.requires_grad:
            state[n] = p.detach().cpu().clone()
    return state


def set_trainable_state(model: nn.Module, state: dict) -> None:
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n in own:
                own[n].copy_(torch.as_tensor(v, dtype=own[n].dtype, device=own[n].device))


def state_to_vector(state: dict, names: list[str]) -> np.ndarray:
    chunks = [np.asarray(state[n]).ravel() for n in names]
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def vector_to_state(vec: np.ndarray, ref_state: dict, names: list[str]) -> "OrderedDict":
    out = OrderedDict()
    offset = 0
    for n in names:
        shape = np.asarray(ref_state[n]).shape
        size = int(np.prod(shape)) if shape else 1
        out[n] = vec[offset : offset + size].reshape(shape)
        offset += size
    return out


def state_diff(new_state: dict, old_state: dict) -> "OrderedDict":
    out = OrderedDict()
    for n, v in new_state.items():
        out[n] = np.asarray(v, dtype=np.float64) - np.asarray(old_state[n], dtype=np.float64)
    return out
