#!/usr/bin/env python
# Download + partition a TDC dataset.  Run: python scripts/prepare_data.py --config configs/caco2.yaml
import argparse

from hcfl.config import HCFLConfig
from hcfl.engine.context import prepare_and_save
from hcfl.utils import get_logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--clients", type=int, default=None)
    args = p.parse_args()

    cfg = HCFLConfig.load(args.config) if args.config else HCFLConfig()
    if args.dataset:
        cfg.data.name = args.dataset
    if args.alpha is not None:
        cfg.data.dirichlet_alpha = args.alpha
    if args.clients is not None:
        cfg.federated.num_clients = args.clients

    log = get_logger("hcfl.prepare")
    summary = prepare_and_save(cfg)
    log.info(f"Prepared dataset -> {summary}")


if __name__ == "__main__":
    main()
