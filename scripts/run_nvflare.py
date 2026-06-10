#!/usr/bin/env python
# Run HCFL as an NVFlare job.  Run: python scripts/run_nvflare.py --config configs/caco2.yaml --threads 4 --gpu 0
import argparse

from hcfl.integration.job import run_job


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workspace", default="/tmp/hcfl_nvflare")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--gpu", default=None, help="comma-separated GPU ids, e.g. '0'")
    p.add_argument("--task-timeout", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=1)
    args = p.parse_args()

    run_job(
        config_path=args.config,
        workspace=args.workspace,
        threads=args.threads,
        gpu=args.gpu,
        task_timeout=args.task_timeout,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
