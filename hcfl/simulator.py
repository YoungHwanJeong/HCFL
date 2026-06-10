# Single-process HCFL runner (driven by scripts/run_simulation.py).
from __future__ import annotations

import json
import os

import numpy as np

from hcfl.config import HCFLConfig
from hcfl.engine.context import load_context
from hcfl.engine.server import HCFLServer
from hcfl.engine.trainer import LocalTrainer
from hcfl.metrics import jains_fairness_index, t90, completion_ratio
from hcfl.models.chemberta import build_model
from hcfl.models.embedder import MoleculeEmbedder
from hcfl.utils import set_seed, get_logger


def run_simulation(cfg: HCFLConfig, eval_every: int = 1, verbose: bool = True) -> dict:
    log = get_logger("hcfl.sim")
    set_seed(cfg.seed)
    device = cfg.device if _cuda_ok(cfg.device) else "cpu"

    ctx = load_context(cfg)
    log.info(f"Loaded {cfg.data.name}: {len(ctx.partitions)} clients, "
             f"task={ctx.spec.task}, classes={ctx.spec.num_classes}")

    server_model = build_model(cfg.model, ctx.spec)
    client_model = build_model(cfg.model, ctx.spec)
    embedder = MoleculeEmbedder(cfg.model.embedder_name, cfg.data.max_smiles_len, device)

    server = HCFLServer(cfg, ctx, server_model, device=device)
    trainer = LocalTrainer(client_model, ctx, embedder, cfg, device=device)

    test_loader = ctx.test_loader()
    metric_key = "accuracy" if ctx.spec.task == "classification" else "mse"
    history, round_logs = [], []
    rng = np.random.default_rng(cfg.seed)

    for t in range(1, cfg.federated.num_rounds + 1):
        assignments = server.start_round(t)
        global_state = server.global_trainable_state()
        results = {}
        for a in assignments:
            results[a.sub_id] = trainer.train(
                global_state, a.shard_indices, a.sub_id, server.profiles[a.parent], rng=rng
            )
        info = server.aggregate_and_update(t, results, cfg.federated.deadline)

        if t % eval_every == 0 or t == cfg.federated.num_rounds:
            m = server.evaluate(test_loader)
            history.append(m[metric_key])
            round_logs.append({"round": t, **m, "cr": info.completion_ratio,
                               "n_sel": len(info.selected), "n_done": len(info.completed),
                               "alpha": round(info.alpha, 3), "splits": len(info.split_parents)})
            if verbose:
                log.info(f"[r{t:03d}] {metric_key}={m[metric_key]:.4f} "
                         f"sel={len(info.selected)} done={len(info.completed)} "
                         f"CR={info.completion_ratio:.2f} alpha={info.alpha:.2f}")

    summary = {
        "dataset": cfg.data.name,
        "task": ctx.spec.task,
        "estimator": cfg.contribution.estimator,
        "final_" + metric_key: history[-1] if history else None,
        "best_" + metric_key: (max(history) if ctx.spec.task == "classification" else min(history)) if history else None,
        "t90": t90(history, ctx.spec.task),
        "jfi": jains_fairness_index(server.selection_counts),
        "completion_ratio": completion_ratio(server.per_round_completion),
        "history": history,
        "round_logs": round_logs,
        "config": cfg.to_dict(),
    }
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"{cfg.data.name}_{cfg.contribution.estimator}_a{cfg.data.dirichlet_alpha}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved results -> {out_path}")
    return summary


def _cuda_ok(device: str) -> bool:
    if device != "cuda":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
