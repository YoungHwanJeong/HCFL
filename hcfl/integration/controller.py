# NVFlare server controller: per-round selection, broadcast, collect, aggregate, VCS.
from __future__ import annotations

import json
import os

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from hcfl.config import HCFLConfig
from hcfl.engine.context import load_context
from hcfl.engine.server import HCFLServer
from hcfl.engine.trainer import LocalResult
from hcfl.integration.executor import TRAIN_TASK
from hcfl.integration.naming import site_name
from hcfl.metrics import completion_ratio, jains_fairness_index, t90
from hcfl.models.chemberta import build_model
from hcfl.models.params import state_to_vector
from hcfl.serial import bytes_to_ndarray, ndarray_to_bytes
from hcfl.utils import get_logger, set_seed


class HCFLController(Controller):
    def __init__(self, config_path: str, task_timeout: int = 600, eval_every: int = 1):
        super().__init__()
        self.config_path = config_path
        self.task_timeout = task_timeout
        self.eval_every = eval_every
        self.log = get_logger("hcfl.controller")

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.cfg = HCFLConfig.load(self.config_path)
        set_seed(self.cfg.seed)
        self.device = _device(self.cfg.device)
        self.ctx = load_context(self.cfg)
        self.model = build_model(self.cfg.model, self.ctx.spec)
        self.server = HCFLServer(self.cfg, self.ctx, self.model, device=self.device)
        self.names = self.server.names
        self.test_loader = self.ctx.test_loader()
        self.metric_key = "accuracy" if self.ctx.spec.task == "classification" else "mse"
        self.history, self.round_logs = [], []
        self.log.info(f"Controller started: {self.cfg.data.name}, "
                      f"{len(self.ctx.partitions)} parents, device={self.device}")

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self._save_summary()

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        for t in range(1, self.cfg.federated.num_rounds + 1):
            if abort_signal.triggered:
                return
            assignments = self.server.start_round(t)
            if not assignments:
                continue

            payload = Shareable()
            state = self.server.global_trainable_state()
            payload["weights"] = ndarray_to_bytes(state_to_vector(state, self.names))
            payload["round"] = t
            payload["deadline"] = self.cfg.federated.deadline
            payload["assignments"] = {
                site_name(a.parent): {
                    "sub_id": a.sub_id,
                    "parent": int(a.parent),
                    "shard_indices": [int(i) for i in a.shard_indices],
                }
                for a in assignments
            }
            targets = [site_name(a.parent) for a in assignments]

            self._round_results: dict[str, LocalResult] = {}
            task = Task(name=TRAIN_TASK, data=payload, timeout=self.task_timeout,
                        result_received_cb=self._on_result)
            self.broadcast_and_wait(
                task=task, fl_ctx=fl_ctx, targets=targets,
                min_responses=len(targets), wait_time_after_min_received=0,
                abort_signal=abort_signal,
            )

            info = self.server.aggregate_and_update(t, self._round_results, self.cfg.federated.deadline)

            if t % self.eval_every == 0 or t == self.cfg.federated.num_rounds:
                m = self.server.evaluate(self.test_loader)
                self.history.append(m[self.metric_key])
                self.round_logs.append({"round": t, **m, "cr": info.completion_ratio,
                                        "n_sel": len(info.selected), "n_done": len(info.completed),
                                        "alpha": round(info.alpha, 3)})
                self.log.info(f"[r{t:03d}] {self.metric_key}={m[self.metric_key]:.4f} "
                              f"sel={len(info.selected)} done={len(info.completed)} "
                              f"CR={info.completion_ratio:.2f}")

    def _on_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result: Shareable = client_task.result
        if result is None or not result.get("selected", False):
            return
        try:
            res = LocalResult(
                sub_id=result["sub_id"],
                delta_vec=bytes_to_ndarray(result["delta"]),
                embeddings=bytes_to_ndarray(result["embeddings"]),
                local_gain=float(result["local_gain"]),
                runtime=float(result["runtime"]),
                n_samples=int(result["n_samples"]),
            )
            self._round_results[res.sub_id] = res
        except Exception as e:
            self.log.error(f"failed to parse result from {client_task.client.name}: {e}")

    def _save_summary(self) -> None:
        if not getattr(self, "history", None):
            return
        summary = {
            "dataset": self.cfg.data.name,
            "task": self.ctx.spec.task,
            "estimator": self.cfg.contribution.estimator,
            "final_" + self.metric_key: self.history[-1],
            "t90": t90(self.history, self.ctx.spec.task),
            "jfi": jains_fairness_index(self.server.selection_counts),
            "completion_ratio": completion_ratio(self.server.per_round_completion),
            "history": self.history,
            "round_logs": self.round_logs,
            "config": self.cfg.to_dict(),
        }
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = os.path.join(self.cfg.output_dir,
                            f"nvflare_{self.cfg.data.name}_{self.cfg.contribution.estimator}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.log.info(f"Saved NVFlare run summary -> {path}")


def _device(want: str) -> str:
    if want == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"
