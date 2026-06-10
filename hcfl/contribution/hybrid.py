# Hybrid score H_k = alpha*sigma(P_k) + (1-alpha)*psi(C_k) with dynamic alpha.
from __future__ import annotations

import numpy as np


def normalize_minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.array(list(values.values()), dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: float((values[k] - lo) / (hi - lo)) for k in values}


def normalize_rank(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    keys = list(values.keys())
    arr = np.array([values[k] for k in keys], dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr))
    if len(arr) > 1:
        ranks /= (len(arr) - 1)
    else:
        ranks[:] = 0.5
    return {k: float(r) for k, r in zip(keys, ranks)}


def alpha_schedule(round_t: int, total_rounds: int, cfg) -> float:
    if not cfg.adapt_alpha or total_rounds <= 1:
        return float(cfg.alpha_init)
    frac = min(max(round_t / (total_rounds - 1), 0.0), 1.0)
    return float(cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * frac)


class HybridContribution:
    def __init__(self, perf_norm: str = "rank", cov_norm: str = "minmax"):
        self._perf_norm = normalize_rank if perf_norm == "rank" else normalize_minmax
        self._cov_norm = normalize_rank if cov_norm == "rank" else normalize_minmax

    def combine(
        self,
        perf_scores: dict[str, float],
        cov_scores: dict[str, float],
        alpha: float,
    ) -> dict[str, float]:
        keys = set(perf_scores) | set(cov_scores)
        p = {k: perf_scores.get(k, 0.0) for k in keys}
        c = {k: cov_scores.get(k, 0.0) for k in keys}
        sp = self._perf_norm(p)
        pc = self._cov_norm(c)
        return {k: float(alpha * sp.get(k, 0.5) + (1 - alpha) * pc.get(k, 0.5)) for k in keys}
