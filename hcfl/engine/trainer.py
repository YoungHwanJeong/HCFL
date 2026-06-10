# Client-side local training: returns update, embeddings, gain, simulated runtime.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hcfl.models.chemberta import task_loss_fn
from hcfl.models.params import (
    trainable_names, get_trainable_state, set_trainable_state, state_to_vector, state_diff,
)
from hcfl.selection.system_sim import estimate_runtime


@dataclass
class LocalResult:
    sub_id: str
    delta_vec: np.ndarray
    embeddings: np.ndarray
    local_gain: float
    runtime: float
    n_samples: int


class LocalTrainer:
    def __init__(self, model, ctx, embedder, cfg, device: str = "cpu"):
        self.model = model
        self.ctx = ctx
        self.embedder = embedder
        self.cfg = cfg
        self.device = device
        self.loss_fn = task_loss_fn(ctx.spec.task)
        self.names = trainable_names(model)

    @torch.no_grad()
    def _eval_loss(self, loader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            b = _to_device(batch, self.device)
            out = self.model(b)
            loss = self.loss_fn(out, b["label"])
            total += float(loss) * b["label"].shape[0]
            n += b["label"].shape[0]
        return total / max(n, 1)

    def train(self, global_state: dict, shard_indices, sub_id: str, profile, rng=None) -> LocalResult:
        set_trainable_state(self.model, global_state)
        self.model.to(self.device)
        before = get_trainable_state(self.model)

        loader = self.ctx.client_loader(shard_indices, shuffle=True)
        loss_before = self._eval_loss(loader)

        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.federated.learning_rate,
        )
        self.model.train()
        for _ in range(self.cfg.federated.local_epochs):
            for batch in loader:
                b = _to_device(batch, self.device)
                opt.zero_grad()
                out = self.model(b)
                loss = self.loss_fn(out, b["label"])
                loss.backward()
                opt.step()

        loss_after = self._eval_loss(loader)
        after = get_trainable_state(self.model)
        delta = state_diff(after, before)
        delta_vec = state_to_vector(delta, self.names).astype(np.float32)

        smiles = self.ctx.smiles_for(shard_indices)
        embeddings = self.embedder.embed(smiles)

        runtime = estimate_runtime(profile, len(shard_indices), self.cfg.federated.local_epochs, rng=rng)
        return LocalResult(
            sub_id=sub_id,
            delta_vec=delta_vec,
            embeddings=embeddings.astype(np.float32),
            local_gain=float(loss_before - loss_after),
            runtime=runtime,
            n_samples=len(shard_indices),
        )


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)
    return out
