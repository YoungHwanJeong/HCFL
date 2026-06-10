# Server engine: one HCFL round (selection, scoring, aggregation, VCS).
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from hcfl.contribution.coverage import CoverageContribution, GlobalDensity
from hcfl.contribution.estimators import EvalContext, build_estimator
from hcfl.contribution.hybrid import HybridContribution, alpha_schedule
from hcfl.contribution.performance import PerformanceContribution
from hcfl.models.chemberta import task_loss_fn
from hcfl.models.params import (
    trainable_names, get_trainable_state, set_trainable_state,
    state_to_vector, vector_to_state,
)
from hcfl.selection.adaptive import AdaptiveSelector
from hcfl.selection.system_sim import sample_profiles
from hcfl.selection.vcs import VCSManager


@dataclass
class Assignment:
    sub_id: str
    parent: int
    shard_indices: np.ndarray


@dataclass
class RoundInfo:
    round: int
    selected: list = field(default_factory=list)
    completed: list = field(default_factory=list)
    stragglers: list = field(default_factory=list)
    hybrid_scores: dict = field(default_factory=dict)
    perf_scores: dict = field(default_factory=dict)
    cov_scores: dict = field(default_factory=dict)
    alpha: float = 0.5
    completion_ratio: float = 0.0
    split_parents: list = field(default_factory=list)


class HCFLServer:
    def __init__(self, cfg, ctx, model, device: str = "cpu"):
        self.cfg = cfg
        self.ctx = ctx
        self.model = model.to(device)
        self.device = device
        self.names = trainable_names(model)
        self.loss_fn = task_loss_fn(ctx.spec.task)

        self.vcs = VCSManager(ctx.partitions, cfg.selection, seed=cfg.seed)
        self.selector = AdaptiveSelector(self.vcs, cfg.selection, seed=cfg.seed)
        self.profiles = {k: p for k, p in
                         zip(ctx.partitions, sample_profiles(len(ctx.partitions), cfg.federated, cfg.seed))}

        self.coverage = CoverageContribution(
            chi2_quantile=cfg.contribution.chi2_quantile,
            sparsity_power=cfg.contribution.sparsity_power,
            qmc_samples=cfg.contribution.qmc_samples,
            cov_shrinkage=cfg.contribution.cov_shrinkage,
            shape_penalty_weight=cfg.contribution.shape_penalty_weight,
            vol_penalty_weight=cfg.contribution.vol_penalty_weight,
            eps=cfg.contribution.eps, seed=cfg.seed,
        )
        self.hybrid = HybridContribution(perf_norm="rank", cov_norm="minmax")
        self.estimator = build_estimator(cfg.contribution.estimator)

        self.selection_counts = {k: 0 for k in ctx.partitions}
        self.per_round_completion: list[float] = []
        self.rng = np.random.default_rng(cfg.seed)

    def global_trainable_state(self) -> dict:
        return get_trainable_state(self.model)

    def num_select(self) -> int:
        return max(1, int(round(self.cfg.federated.participation_rate * len(self.ctx.partitions))))

    def start_round(self, t: int) -> list[Assignment]:
        self.selector.update_priorities()
        selected = self.selector.select(self.num_select())
        assignments = []
        for sid in selected:
            parent = self.vcs.parent_of(sid)
            shard = self.vcs.shard_of(sid)
            assignments.append(Assignment(sub_id=sid, parent=parent, shard_indices=shard))
        self._last_selected = selected
        return assignments

    def aggregate_and_update(self, t: int, results: dict, deadline: float) -> RoundInfo:
        selected = list(getattr(self, "_last_selected", list(results)))
        info = RoundInfo(round=t, selected=selected)

        completed, stragglers = {}, []
        for sid in selected:
            res = results.get(sid)
            if res is not None and res.runtime <= deadline and res.delta_vec.size > 0:
                completed[sid] = res
            else:
                stragglers.append(sid)
        info.completed = list(completed)
        info.stragglers = stragglers
        info.completion_ratio = len(completed) / max(len(selected), 1)
        self.per_round_completion.append(info.completion_ratio)

        if completed:
            deltas = {sid: r.delta_vec.astype(np.float64) for sid, r in completed.items()}
            embeddings = {sid: r.embeddings for sid, r in completed.items()}
            local_gains = {sid: r.local_gain for sid, r in completed.items()}

            info.cov_scores = self._coverage_scores(embeddings)
            info.perf_scores = self._performance_scores(deltas, local_gains)
            info.alpha = alpha_schedule(t, self.cfg.federated.num_rounds, self.cfg.contribution)
            info.hybrid_scores = self.hybrid.combine(info.perf_scores, info.cov_scores, info.alpha)
            self._apply_aggregation(deltas, info.hybrid_scores)

        self.selector.update_after_round(selected, info.hybrid_scores)
        for sid in completed:
            self.selection_counts[self.vcs.parent_of(sid)] += 1

        straggled_parents = {self.vcs.parent_of(sid) for sid in stragglers}
        for sid in selected:
            parent = self.vcs.parent_of(sid)
            self.vcs.record_outcome(parent, straggled=(parent in straggled_parents))
        info.split_parents = self.vcs.maybe_split_all()
        if info.split_parents:
            self.selector.resync()
        return info

    def _coverage_scores(self, embeddings: dict) -> dict:
        pooled = np.concatenate([z for z in embeddings.values() if z.shape[0] > 0], axis=0) \
            if any(z.shape[0] > 0 for z in embeddings.values()) else np.zeros((0, 1))
        if pooled.shape[0] < 2:
            return {sid: 0.0 for sid in embeddings}
        density = GlobalDensity(
            pooled.astype(np.float64),
            threshold_quantile=self.cfg.contribution.density_threshold_quantile,
            seed=self.cfg.seed,
        )
        return self.coverage.scores({sid: z.astype(np.float64) for sid, z in embeddings.items()}, density)

    def _performance_scores(self, deltas: dict, local_gains: dict) -> dict:
        perf_contrib = None
        if self.cfg.contribution.estimator == "hybrid":
            perf_contrib = PerformanceContribution(
                self.model, self.loss_fn, device=self.device,
                max_samples=self.cfg.contribution.fim_batch_size * self.cfg.contribution.fim_max_batches,
                batch_size=self.cfg.contribution.fim_batch_size,
            )
            perf_contrib.fit_reference(self.ctx.reference_loader())
        ctx = EvalContext(
            deltas=deltas, local_gains=local_gains,
            utility_fn=self._make_utility_fn(deltas) if self.cfg.contribution.estimator in
            {"loo", "shap", "leastcore"} else None,
            perf_contrib=perf_contrib, seed=self.cfg.seed,
        )
        return self.estimator.score(ctx)

    def _make_utility_fn(self, deltas: dict):
        base_state = get_trainable_state(self.model)
        base_vec = state_to_vector(base_state, self.names)
        ref_loader = self.ctx.reference_loader()

        def utility(subset_ids):
            subset = [s for s in subset_ids if s in deltas]
            if not subset:
                agg = np.zeros_like(base_vec)
            else:
                agg = np.mean([deltas[s] for s in subset], axis=0)
            new_vec = base_vec + agg
            set_trainable_state(self.model, vector_to_state(new_vec, base_state, self.names))
            loss = self._reference_loss(ref_loader)
            set_trainable_state(self.model, base_state)
            return -loss

        return utility

    @torch.no_grad()
    def _reference_loss(self, loader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            b = _to_device(batch, self.device)
            out = self.model(b)
            loss = self.loss_fn(out, b["label"])
            total += float(loss) * b["label"].shape[0]
            n += b["label"].shape[0]
        return total / max(n, 1)

    def _apply_aggregation(self, deltas: dict, hybrid_scores: dict) -> None:
        base_state = get_trainable_state(self.model)
        base_vec = state_to_vector(base_state, self.names)
        weights = np.array([max(hybrid_scores.get(sid, 0.0), 0.0) for sid in deltas], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(deltas))
        weights /= weights.sum()
        agg = np.zeros_like(base_vec)
        for w, dv in zip(weights, deltas.values()):
            agg += w * dv
        new_vec = base_vec + agg
        set_trainable_state(self.model, vector_to_state(new_vec, base_state, self.names))

    def evaluate(self, loader) -> dict:
        from hcfl.metrics import evaluate_model
        return evaluate_model(self.model, loader, self.ctx.spec.task, self.device)


def _to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)
    return out
