# ChemBERTa backbone + task head (regression / DDI classification).
from __future__ import annotations

import torch
import torch.nn as nn


def build_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class ChemBERTaClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        task: str = "regression",
        num_classes: int = 1,
        is_pair: bool = False,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        freeze_backbone_layers: int = 0,
    ):
        super().__init__()
        from transformers import AutoModel

        self.task = task
        self.is_pair = is_pair
        self.num_classes = num_classes

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size

        in_dim = hidden * (4 if is_pair else 1)
        out_dim = num_classes if task == "classification" else 1

        self.head = nn.Sequential(
            nn.Linear(in_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, out_dim),
        )

        self._maybe_freeze(freeze_backbone_layers)

    def _maybe_freeze(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        enc_layers = self.backbone.encoder.layer
        for layer in enc_layers[: min(n_layers, len(enc_layers))]:
            for p in layer.parameters():
                p.requires_grad = False

    def encode(self, tokens: dict) -> torch.Tensor:
        out = self.backbone(**tokens)
        return _mean_pool(out.last_hidden_state, tokens["attention_mask"])

    def forward(self, batch: dict) -> torch.Tensor:
        h_a = self.encode(batch["a"])
        if self.is_pair:
            h_b = self.encode(batch["b"])
            feats = torch.cat([h_a, h_b, (h_a - h_b).abs(), h_a * h_b], dim=-1)
        else:
            feats = h_a
        logits = self.head(feats)
        if self.task == "regression":
            return logits.squeeze(-1)
        return logits


def build_model(model_cfg, data_spec) -> ChemBERTaClassifier:
    return ChemBERTaClassifier(
        backbone_name=model_cfg.backbone_name,
        task=data_spec.task,
        num_classes=data_spec.num_classes,
        is_pair=data_spec.is_pair,
        head_hidden_dim=model_cfg.head_hidden_dim,
        head_dropout=model_cfg.head_dropout,
        freeze_backbone_layers=model_cfg.freeze_backbone_layers,
    )


def task_loss_fn(task: str):
    if task == "classification":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()
