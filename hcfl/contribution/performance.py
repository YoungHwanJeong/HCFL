# Performance-based contribution P_k via Taylor expansion + Fisher diagonal.
from __future__ import annotations

import numpy as np
import torch

from hcfl.models.params import trainable_names


class PerformanceContribution:
    def __init__(self, model, loss_fn, device="cpu", max_samples=256, batch_size=32, eps=1e-8):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.eps = eps

        self.names = trainable_names(model)
        self.g: np.ndarray | None = None
        self.f_diag: np.ndarray | None = None

    def _flat_grad(self) -> np.ndarray:
        return np.concatenate(
            [
                p.grad.detach().reshape(-1).cpu().numpy().astype(np.float64)
                if p.grad is not None
                else np.zeros(p.numel(), dtype=np.float64)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            ]
        )

    @torch.enable_grad()
    def fit_reference(self, reference_loader) -> None:
        self.model.to(self.device).eval()
        d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        g_sum = np.zeros(d, dtype=np.float64)
        f_sum = np.zeros(d, dtype=np.float64)
        n_seen = 0

        for batch in reference_loader:
            if n_seen >= self.max_samples:
                break
            batch = _to_device(batch, self.device)
            bsz = batch["label"].shape[0]
            for i in range(bsz):
                if n_seen >= self.max_samples:
                    break
                single = _index_batch(batch, i)
                self.model.zero_grad(set_to_none=True)
                out = self.model(single)
                loss = self.loss_fn(out, single["label"].unsqueeze(0)
                                    if single["label"].dim() == 0 else single["label"])
                loss.backward()
                gi = self._flat_grad()
                g_sum += gi
                f_sum += gi * gi
                n_seen += 1

        n = max(n_seen, 1)
        self.g = g_sum / n
        self.f_diag = f_sum / n
        self.model.zero_grad(set_to_none=True)

    def score(self, delta_vec: np.ndarray) -> float:
        if self.g is None or self.f_diag is None:
            raise RuntimeError("Call fit_reference() before score().")
        linear = float(self.g @ delta_vec)
        quad = float(np.sum(self.f_diag * delta_vec * delta_vec))
        return -linear - 0.5 * quad


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v.to(device)
    return out


def _index_batch(batch: dict, i: int) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv[i : i + 1] for kk, vv in v.items()}
        else:
            out[k] = v[i : i + 1]
    return out
