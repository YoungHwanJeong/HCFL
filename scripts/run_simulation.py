#!/usr/bin/env python
# Run HCFL standalone.  Run: python scripts/run_simulation.py --config configs/caco2.yaml
import argparse

from hcfl.config import HCFLConfig
from hcfl.simulator import run_simulation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--estimator", default=None,
                   choices=["hybrid", "individual", "loo", "shap", "leastcore"])
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--clients", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = p.parse_args()

    cfg = HCFLConfig.load(args.config) if args.config else HCFLConfig()
    if args.dataset:
        cfg.data.name = args.dataset
    if args.alpha is not None:
        cfg.data.dirichlet_alpha = args.alpha
    if args.estimator:
        cfg.contribution.estimator = args.estimator
    if args.rounds is not None:
        cfg.federated.num_rounds = args.rounds
    if args.clients is not None:
        cfg.federated.num_clients = args.clients
    if args.device:
        cfg.device = args.device

    summary = run_simulation(cfg)
    print("\n==== HCFL summary ====")
    for k in ("dataset", "estimator", "t90", "jfi", "completion_ratio"):
        print(f"  {k}: {summary[k]}")
    final_key = [k for k in summary if k.startswith("final_")][0]
    print(f"  {final_key}: {summary[final_key]:.4f}")


if __name__ == "__main__":
    main()
