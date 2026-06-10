# Build and run the HCFL NVFlare job (driven by scripts/run_nvflare.py).
from __future__ import annotations

import os

from nvflare.job_config.api import FedJob

from hcfl.config import HCFLConfig
from hcfl.integration.controller import HCFLController
from hcfl.integration.executor import HCFLExecutor, TRAIN_TASK


def build_job(config_path: str, task_timeout: int = 600, eval_every: int = 1) -> FedJob:
    config_path = os.path.abspath(config_path)
    cfg = HCFLConfig.load(config_path)

    job = FedJob(name=f"hcfl_{cfg.data.name}_{cfg.contribution.estimator}")

    controller = HCFLController(config_path=config_path, task_timeout=task_timeout, eval_every=eval_every)
    job.to_server(controller)

    executor = HCFLExecutor(config_path=config_path)
    job.to_clients(executor, tasks=[TRAIN_TASK])
    return job


def run_job(
    config_path: str,
    workspace: str = "/tmp/hcfl_nvflare",
    threads: int | None = None,
    gpu: str | None = None,
    task_timeout: int = 600,
    eval_every: int = 1,
) -> None:
    cfg = HCFLConfig.load(os.path.abspath(config_path))
    cfg.data.data_root = os.path.abspath(cfg.data.data_root)
    cfg.output_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(workspace, exist_ok=True)
    resolved_path = os.path.join(cfg.data.data_root, "_hcfl_run_config.yaml")
    cfg.save(resolved_path)

    n_clients = cfg.federated.num_clients
    threads = threads or min(n_clients, 4)
    job = build_job(resolved_path, task_timeout=task_timeout, eval_every=eval_every)
    job.simulator_run(workspace, n_clients=n_clients, threads=threads, gpu=gpu)
