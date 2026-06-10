# Evaluation metrics: MSE/Accuracy, T90, Jain's Fairness Index, Completion Ratio.
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate_model(model, loader, task: str, device: str = "cpu") -> dict:
    model.to(device).eval()
    preds, targets = [], []
    for batch in loader:
        b = {}
        for k, v in batch.items():
            b[k] = {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)
        out = model(b)
        if task == "classification":
            preds.append(out.argmax(dim=-1).cpu())
        else:
            preds.append(out.cpu())
        targets.append(batch["label"])
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if task == "classification":
        return {"accuracy": float((preds == targets).mean()), "n": len(targets)}
    mse = float(np.mean((preds - targets) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "n": len(targets)}


def jains_fairness_index(selection_counts: dict | list) -> float:
    x = np.asarray(list(selection_counts.values()) if isinstance(selection_counts, dict)
                   else selection_counts, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    return float((x.sum() ** 2) / (x.size * np.sum(x ** 2)))


def t90(history: list[float], task: str) -> int | None:
    if not history:
        return None
    h = np.asarray(history, dtype=np.float64)
    if task == "classification":
        thresh = 0.9 * h[-1]
        hit = np.where(h >= thresh)[0]
    else:
        s0, sf = h[0], h[-1]
        thresh = s0 - 0.9 * (s0 - sf)
        hit = np.where(h <= thresh)[0]
    return int(hit[0]) if len(hit) else None


def completion_ratio(per_round_completion: list[float]) -> float:
    return float(np.mean(per_round_completion)) if per_round_completion else 0.0
