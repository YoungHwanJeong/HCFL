# Virtual Client Splitting: split persistent stragglers into IID sub-clients.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParentSplitState:
    parent: int
    indices: np.ndarray
    m: int = 1
    shards: list = field(default_factory=list)
    consecutive_failures: int = 0
    failure_ema: float = 0.0
    splits_done: int = 0

    def sub_ids(self) -> list[str]:
        return [f"{self.parent}_s{j}" for j in range(self.m)]


class VCSManager:
    def __init__(self, parent_indices: dict[int, np.ndarray], cfg_selection, seed: int = 42):
        self.cfg = cfg_selection
        self.rng = np.random.default_rng(seed)
        self.states: dict[int, ParentSplitState] = {}
        for k, idx in parent_indices.items():
            idx = np.asarray(idx, dtype=int)
            self.states[k] = ParentSplitState(parent=k, indices=idx, m=1, shards=[idx.copy()])

    def parent_of(self, sub_id: str) -> int:
        return int(sub_id.split("_s")[0])

    def shard_of(self, sub_id: str) -> np.ndarray:
        parent, j = sub_id.split("_s")
        return self.states[int(parent)].shards[int(j)]

    def sub_ids_of(self, parent: int) -> list[str]:
        return self.states[parent].sub_ids()

    def record_outcome(self, parent: int, straggled: bool) -> None:
        st = self.states[parent]
        st.consecutive_failures = st.consecutive_failures + 1 if straggled else 0
        g = self.cfg.failure_ema
        st.failure_ema = g * (1.0 if straggled else 0.0) + (1 - g) * st.failure_ema

    def should_split(self, parent: int) -> bool:
        st = self.states[parent]
        if st.splits_done >= self.cfg.max_splits:
            return False
        if len(st.indices) < 2 * (st.m + 1):
            return False
        return (st.consecutive_failures >= self.cfg.r_split) or (st.failure_ema > self.cfg.tau_split)

    def split(self, parent: int) -> int:
        st = self.states[parent]
        new_m = min(st.m * 2, 2 ** self.cfg.max_splits, len(st.indices))
        new_m = max(new_m, st.m)
        if new_m == st.m:
            return st.m
        idx = st.indices.copy()
        self.rng.shuffle(idx)
        st.shards = [np.array(sorted(s), dtype=int) for s in np.array_split(idx, new_m)]
        st.m = new_m
        st.splits_done += 1
        st.consecutive_failures = 0
        return st.m

    def maybe_split_all(self) -> list[int]:
        split_parents = []
        for k in list(self.states):
            if self.should_split(k):
                self.split(k)
                split_parents.append(k)
        return split_parents
