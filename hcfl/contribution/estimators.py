# Performance-signal estimators: hybrid / individual / LOO / SHAP / least-core.
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np


@dataclass
class EvalContext:
    deltas: dict[str, np.ndarray]
    local_gains: Optional[dict[str, float]] = None
    utility_fn: Optional[Callable[[Iterable[str]], float]] = None
    perf_contrib: object = None
    seed: int = 42


class ContribEstimator:
    name = "base"

    def score(self, ctx: EvalContext) -> dict[str, float]:
        raise NotImplementedError


class HybridPerfEstimator(ContribEstimator):
    name = "hybrid"

    def score(self, ctx: EvalContext) -> dict[str, float]:
        if ctx.perf_contrib is None:
            raise RuntimeError("hybrid estimator requires a fitted PerformanceContribution")
        return {cid: ctx.perf_contrib.score(dv) for cid, dv in ctx.deltas.items()}


class IndividualEstimator(ContribEstimator):
    name = "individual"

    def score(self, ctx: EvalContext) -> dict[str, float]:
        if not ctx.local_gains:
            return {cid: -float(np.linalg.norm(dv)) for cid, dv in ctx.deltas.items()}
        return {cid: float(ctx.local_gains.get(cid, 0.0)) for cid in ctx.deltas}


class LOOEstimator(ContribEstimator):
    name = "loo"

    def score(self, ctx: EvalContext) -> dict[str, float]:
        assert ctx.utility_fn is not None, "LOO requires utility_fn"
        all_ids = list(ctx.deltas)
        u_full = ctx.utility_fn(all_ids)
        out = {}
        for cid in all_ids:
            subset = [c for c in all_ids if c != cid]
            out[cid] = float(u_full - ctx.utility_fn(subset))
        return out


class ShapleyEstimator(ContribEstimator):
    name = "shap"

    def __init__(self, num_perm: int = 16):
        self.num_perm = num_perm

    def score(self, ctx: EvalContext) -> dict[str, float]:
        assert ctx.utility_fn is not None, "SHAP requires utility_fn"
        ids = list(ctx.deltas)
        n = len(ids)
        rng = np.random.default_rng(ctx.seed)
        phi = {cid: 0.0 for cid in ids}
        if n <= 6:
            perms = list(itertools.permutations(ids))
        else:
            perms = [list(rng.permutation(ids)) for _ in range(self.num_perm)]
        for perm in perms:
            prev_u = ctx.utility_fn([])
            coalition: list[str] = []
            for cid in perm:
                coalition.append(cid)
                u = ctx.utility_fn(coalition)
                phi[cid] += (u - prev_u)
                prev_u = u
        for cid in ids:
            phi[cid] /= len(perms)
        return phi


class LeastCoreEstimator(ContribEstimator):
    name = "leastcore"

    def __init__(self, num_coalitions: int = 64):
        self.num_coalitions = num_coalitions

    def score(self, ctx: EvalContext) -> dict[str, float]:
        from scipy.optimize import linprog

        assert ctx.utility_fn is not None, "Least-core requires utility_fn"
        ids = list(ctx.deltas)
        n = len(ids)
        if n == 0:
            return {}
        rng = np.random.default_rng(ctx.seed)
        u_full = ctx.utility_fn(ids)

        coalitions = [tuple(np.where(rng.random(n) < 0.5)[0]) for _ in range(self.num_coalitions)]
        coalitions += [(i,) for i in range(n)]
        coalitions = list({c for c in coalitions if len(c) > 0})

        c_obj = np.zeros(n + 1)
        c_obj[-1] = 1.0
        A_ub, b_ub = [], []
        for S in coalitions:
            row = np.zeros(n + 1)
            for k in S:
                row[k] = -1.0
            row[-1] = -1.0
            A_ub.append(row)
            b_ub.append(-ctx.utility_fn([ids[k] for k in S]))
        A_eq = np.concatenate([np.ones(n), [0.0]])[None, :]
        b_eq = [u_full]
        bounds = [(None, None)] * n + [(0, None)]
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            return {cid: float(u_full / n) for cid in ids}
        x = res.x[:n]
        return {cid: float(xi) for cid, xi in zip(ids, x)}


CONTRIB_ESTIMATORS = {
    HybridPerfEstimator.name: HybridPerfEstimator,
    IndividualEstimator.name: IndividualEstimator,
    LOOEstimator.name: LOOEstimator,
    ShapleyEstimator.name: ShapleyEstimator,
    LeastCoreEstimator.name: LeastCoreEstimator,
}


def build_estimator(name: str, **kwargs) -> ContribEstimator:
    if name not in CONTRIB_ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Available: {list(CONTRIB_ESTIMATORS)}")
    cls = CONTRIB_ESTIMATORS[name]
    if name == "shap":
        return cls(num_perm=kwargs.get("num_perm", 16))
    if name == "leastcore":
        return cls(num_coalitions=kwargs.get("num_coalitions", 64))
    return cls()
