# NVFlare client executor: train this site's assigned shard, return update+embeddings.
from __future__ import annotations

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

from hcfl.config import HCFLConfig
from hcfl.engine.context import load_context
from hcfl.engine.trainer import LocalTrainer
from hcfl.integration.naming import parent_id
from hcfl.models.chemberta import build_model
from hcfl.models.embedder import MoleculeEmbedder
from hcfl.models.params import vector_to_state, get_trainable_state
from hcfl.selection.system_sim import sample_profiles
from hcfl.serial import bytes_to_ndarray, ndarray_to_bytes
from hcfl.utils import get_logger, set_seed

TRAIN_TASK = "hcfl_train"


class HCFLExecutor(Executor):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self._ready = False
        self.log = get_logger("hcfl.executor")

    def _lazy_init(self, fl_ctx: FLContext):
        if self._ready:
            return
        self.cfg = HCFLConfig.load(self.config_path)
        set_seed(self.cfg.seed)
        self.device = _device(self.cfg.device)
        self.ctx = load_context(self.cfg)
        self.model = build_model(self.cfg.model, self.ctx.spec)
        self.ref_state = get_trainable_state(self.model)
        self.names = list(self.ref_state.keys())
        self.embedder = MoleculeEmbedder(self.cfg.model.embedder_name,
                                         self.cfg.data.max_smiles_len, self.device)
        self.trainer = LocalTrainer(self.model, self.ctx, self.embedder, self.cfg, self.device)
        self.profiles = sample_profiles(len(self.ctx.partitions), self.cfg.federated, self.cfg.seed)
        self.me = parent_id(fl_ctx.get_identity_name())
        self._ready = True
        self.log.info(f"Executor ready on parent {self.me} (device={self.device})")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext,
                abort_signal: Signal) -> Shareable:
        if task_name != TRAIN_TASK:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        try:
            self._lazy_init(fl_ctx)
        except Exception as e:
            self.log.error(f"init failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        assignments = shareable.get("assignments", {})
        my_name = fl_ctx.get_identity_name()
        if my_name not in assignments:
            out = Shareable()
            out["selected"] = False
            return out

        a = assignments[my_name]
        weights = bytes_to_ndarray(shareable["weights"]).astype("float64")
        global_state = vector_to_state(weights, self.ref_state, self.names)
        shard = a["shard_indices"]
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        res = self.trainer.train(global_state, shard, a["sub_id"], self.profiles[self.me])
        out = Shareable()
        out["selected"] = True
        out["sub_id"] = res.sub_id
        out["delta"] = ndarray_to_bytes(res.delta_vec)
        out["embeddings"] = ndarray_to_bytes(res.embeddings)
        out["local_gain"] = float(res.local_gain)
        out["runtime"] = float(res.runtime)
        out["n_samples"] = int(res.n_samples)
        return out


def _device(want: str) -> str:
    if want == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"
