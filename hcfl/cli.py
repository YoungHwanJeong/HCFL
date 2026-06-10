# Console entry points: hcfl-prepare and hcfl-run.
from __future__ import annotations

import argparse

from hcfl.config import HCFLConfig
from hcfl.engine.context import prepare_and_save
from hcfl.utils import get_logger


def _load_cfg(args) -> HCFLConfig:
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
    return cfg


def prepare_data_main() -> None:
    p = argparse.ArgumentParser(description="Download + partition a TDC dataset for HCFL")
    p.add_argument("--config")
    p.add_argument("--dataset")
    p.add_argument("--alpha", type=float)
    p.add_argument("--estimator")
    p.add_argument("--rounds", type=int)
    p.add_argument("--clients", type=int)
    args = p.parse_args()
    cfg = _load_cfg(args)
    log = get_logger("hcfl.prepare")
    summary = prepare_and_save(cfg)
    log.info(f"Prepared {cfg.data.name}: {summary}")


def run_main() -> None:
    p = argparse.ArgumentParser(description="Run HCFL (standalone simulator or NVFlare)")
    p.add_argument("--config")
    p.add_argument("--dataset")
    p.add_argument("--alpha", type=float)
    p.add_argument("--estimator")
    p.add_argument("--rounds", type=int)
    p.add_argument("--clients", type=int)
    p.add_argument("--backend", choices=["standalone", "nvflare"], default="standalone")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--gpu", default=None)
    args = p.parse_args()
    cfg = _load_cfg(args)

    if args.backend == "nvflare":
        import tempfile
        from hcfl.integration.job import run_job

        tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        cfg.save(tmp.name)
        run_job(tmp.name, threads=args.threads, gpu=args.gpu)
    else:
        from hcfl.simulator import run_simulation

        summary = run_simulation(cfg)
        get_logger("hcfl.run").info(
            f"DONE {cfg.data.name}/{cfg.contribution.estimator}: "
            f"t90={summary['t90']} jfi={summary['jfi']:.3f} cr={summary['completion_ratio']:.3f}"
        )


if __name__ == "__main__":
    run_main()
