# Unit tests for HCFL framework-agnostic logic.  Run: pytest tests/ -q
import numpy as np
import pandas as pd
import pytest

from hcfl.config import HCFLConfig
from hcfl.contribution.coverage import CoverageContribution, GlobalDensity, robust_mean_cov
from hcfl.contribution.estimators import EvalContext, build_estimator
from hcfl.contribution.hybrid import HybridContribution, normalize_minmax, normalize_rank, alpha_schedule
from hcfl.data.partition import dirichlet_partition, assign_workloads
from hcfl.metrics import jains_fairness_index, t90, completion_ratio
from hcfl.selection.adaptive import AdaptiveSelector
from hcfl.selection.system_sim import sample_profiles, estimate_runtime
from hcfl.selection.vcs import VCSManager


def test_config_roundtrip(tmp_path):
    cfg = HCFLConfig()
    cfg.data.dirichlet_alpha = 0.123
    p = tmp_path / "c.yaml"
    cfg.save(str(p))
    cfg2 = HCFLConfig.load(str(p))
    assert cfg2.data.dirichlet_alpha == pytest.approx(0.123)
    assert cfg2.contribution.estimator == cfg.contribution.estimator


def test_dirichlet_partition_classification():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"label": rng.integers(0, 5, size=2000)})
    parts = dirichlet_partition(df, num_clients=20, alpha=0.1, task="classification", seed=1)
    total = sum(len(p) for p in parts)
    assert total == len(df)
    flat = np.concatenate([p for p in parts if len(p)])
    assert len(np.unique(flat)) == total


def test_dirichlet_more_skew_with_small_alpha():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"label": rng.integers(0, 5, size=4000)})
    def avg_classes(alpha):
        parts = dirichlet_partition(df, 30, alpha, "classification", seed=2)
        labs = df["label"].to_numpy()
        return np.mean([len(np.unique(labs[p])) for p in parts if len(p) > 0])
    assert avg_classes(0.05) < avg_classes(2.0)


def test_assign_workloads_caps_size():
    parts = [np.arange(1000), np.arange(50)]
    out = assign_workloads(parts, 100, 500, seed=0)
    assert out[0].size <= 500
    assert out[1].size == 50


def test_robust_cov_is_pd():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(100, 8))
    mu, cov = robust_mean_cov(Z)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_coverage_rewards_sparse_region():
    rng = np.random.default_rng(0)
    d = 4
    dense = rng.normal(0, 1, size=(800, d))
    Z_G = np.concatenate([dense, rng.normal(8, 0.5, size=(20, d))])
    density = GlobalDensity(Z_G, threshold_quantile=0.3, seed=0)
    cov = CoverageContribution(qmc_samples=256, seed=0)
    emb = {
        "dense": rng.normal(0, 1, size=(60, d)),
        "sparse": rng.normal(8, 0.4, size=(60, d)),
    }
    scores = cov.scores(emb, density)
    assert scores["sparse"] >= scores["dense"]


def test_normalizers_range():
    vals = {"a": -3.0, "b": 0.0, "c": 5.0}
    for fn in (normalize_minmax, normalize_rank):
        out = fn(vals)
        assert min(out.values()) >= 0 and max(out.values()) <= 1


def test_alpha_schedule_increasing():
    cfg = HCFLConfig().contribution
    assert alpha_schedule(0, 100, cfg) < alpha_schedule(99, 100, cfg)


def test_hybrid_combine_weighting():
    h = HybridContribution()
    perf = {"a": 1.0, "b": 0.0}
    cov = {"a": 0.0, "b": 1.0}
    out1 = h.combine(perf, cov, alpha=1.0)
    assert out1["a"] > out1["b"]
    out0 = h.combine(perf, cov, alpha=0.0)
    assert out0["b"] > out0["a"]


def _toy_ctx():
    value = {"c0": 3.0, "c1": 1.0, "c2": 2.0}
    deltas = {k: np.ones(4) for k in value}
    def util(subset):
        return float(sum(value[s] for s in subset))
    return EvalContext(deltas=deltas, local_gains=value, utility_fn=util, seed=0), value


def test_individual_estimator():
    ctx, value = _toy_ctx()
    out = build_estimator("individual").score(ctx)
    assert out == value


def test_loo_and_shap_rank_consistently():
    ctx, value = _toy_ctx()
    loo = build_estimator("loo").score(ctx)
    shap = build_estimator("shap").score(ctx)
    assert max(loo, key=loo.get) == "c0"
    assert max(shap, key=shap.get) == "c0"


def test_leastcore_runs():
    ctx, _ = _toy_ctx()
    out = build_estimator("leastcore").score(ctx)
    assert set(out) == {"c0", "c1", "c2"}


def _make_managers(n=10, seed=0):
    rng = np.random.default_rng(seed)
    parts = {k: np.arange(k * 100, k * 100 + int(rng.integers(40, 80))) for k in range(n)}
    cfg = HCFLConfig().selection
    vcs = VCSManager(parts, cfg, seed=seed)
    sel = AdaptiveSelector(vcs, cfg, seed=seed)
    return vcs, sel, cfg


def test_vcs_splits_persistent_straggler():
    vcs, sel, cfg = _make_managers()
    for _ in range(cfg.r_split):
        vcs.record_outcome(0, straggled=True)
    assert vcs.should_split(0)
    m = vcs.split(0)
    assert m >= 2
    shards = [vcs.shard_of(s) for s in vcs.sub_ids_of(0)]
    union = np.concatenate(shards)
    assert len(np.unique(union)) == len(vcs.states[0].indices)


def test_selection_exclusive_one_sub_per_parent():
    vcs, sel, cfg = _make_managers()
    vcs.split(0); sel.resync()
    sel.update_priorities()
    chosen = sel.select(5)
    parents = [vcs.parent_of(s) for s in chosen]
    assert len(parents) == len(set(parents))


def test_priority_aging_floor():
    vcs, sel, cfg = _make_managers()
    q0 = dict(sel.priorities())
    sel.update_priorities()
    q1 = sel.priorities()
    assert all(q1[k] >= q0[k] + 1.0 - 1e-9 for k in q0)


def test_runtime_scales_with_workload():
    cfg = HCFLConfig().federated
    prof = sample_profiles(1, cfg, seed=0)[0]
    assert estimate_runtime(prof, 500, 5) > estimate_runtime(prof, 50, 5)


def test_jfi_uniform_is_one():
    assert jains_fairness_index({"a": 5, "b": 5, "c": 5}) == pytest.approx(1.0)
    assert jains_fairness_index({"a": 10, "b": 0, "c": 0}) < 0.5


def test_t90_regression_and_classification():
    assert t90([1.0, 0.5, 0.2, 0.1], "regression") is not None
    assert t90([0.1, 0.5, 0.9, 0.95], "classification") is not None


def test_completion_ratio():
    assert completion_ratio([1.0, 0.5]) == pytest.approx(0.75)
