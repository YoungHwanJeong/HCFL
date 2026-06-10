# Small frozen ChemBERTa used to embed SMILES for coverage scoring.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MoleculeEmbedder(nn.Module):
    def __init__(
        self,
        embedder_name: str = "DeepChem/ChemBERTa-5M-MLM",
        max_len: int = 128,
        device: str = "cpu",
        ldp_noise_std: float = 0.0,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        self.model = AutoModel.from_pretrained(embedder_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_len = max_len
        self.device = device
        self.ldp_noise_std = float(ldp_noise_std)
        self.embed_dim = self.model.config.hidden_size
        self.to(device)

    @torch.no_grad()
    def _encode_batch(self, smiles: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            [str(s) for s in smiles],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled

    @torch.no_grad()
    def embed(self, smiles_items: list, batch_size: int = 64) -> np.ndarray:
        is_pair = len(smiles_items) > 0 and isinstance(smiles_items[0], (tuple, list))
        embs = []
        for start in range(0, len(smiles_items), batch_size):
            chunk = smiles_items[start : start + batch_size]
            if is_pair:
                a = [c[0] for c in chunk]
                b = [c[1] for c in chunk]
                z = 0.5 * (self._encode_batch(a) + self._encode_batch(b))
            else:
                z = self._encode_batch(chunk)
            embs.append(z.cpu().numpy())
        if not embs:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out = np.concatenate(embs, axis=0).astype(np.float32)
        if self.ldp_noise_std > 0:
            rng = np.random.default_rng()
            out = out + rng.normal(0.0, self.ldp_noise_std, size=out.shape).astype(np.float32)
        return out
