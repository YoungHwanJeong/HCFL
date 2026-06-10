# Adaptive two-stage client selection: priority aging + contribution acceleration.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hcfl.selection.vcs import VCSManager


@dataclass
class ClientState:
    parent: int
    q: float = 0.0
    r: dict = field(default_factory=dict)
    hbar: dict = field(default_factory=dict)


class AdaptiveSelector:
    def __init__(self, vcs: VCSManager, cfg_selection, seed: int = 42):
        self.vcs = vcs
        self.cfg = cfg_selection
        self.rng = np.random.default_rng(seed)
        self.states: dict[int, ClientState] = {k: ClientState(parent=k) for k in vcs.states}
        self._sync_substates()

    def _sync_substates(self) -> None:
        for k, st in self.states.items():
            for sid in self.vcs.sub_ids_of(k):
                st.r.setdefault(sid, 0.0)
                st.hbar.setdefault(sid, 0.0)
            valid = set(self.vcs.sub_ids_of(k))
            for sid in list(st.r):
                if sid not in valid:
                    st.r.pop(sid, None)
                    st.hbar.pop(sid, None)

    def parent_effective_score(self, k: int) -> float:
        st = self.states[k]
        return max(st.hbar.values()) if st.hbar else 0.0

    def update_priorities(self) -> None:
        self._sync_substates()
        eff = {k: self.parent_effective_score(k) for k in self.states}
        mean_eff = float(np.mean(list(eff.values()))) if eff else 0.0
        for k, st in self.states.items():
            f = max(0.0, eff[k] - mean_eff)
            st.q += 1.0 + self.cfg.delta * f

    def select(self, num_select: int) -> list[str]:
        parents = list(self.states)
        q = np.array([max(self.states[k].q, 1e-9) for k in parents], dtype=np.float64)
        probs = q / q.sum()
        n = min(num_select, len(parents))
        chosen_parents = self.rng.choice(parents, size=n, replace=False, p=probs)

        selected = []
        for k in chosen_parents:
            st = self.states[k]
            sid = max(st.r.items(), key=lambda kv: kv[1])[0]
            selected.append(sid)
        return selected

    def update_after_round(self, selected: list[str], hybrid_scores: dict[str, float]) -> None:
        g = self.cfg.ema_gamma
        sel_set = set(selected)
        for sid, h in hybrid_scores.items():
            k = self.vcs.parent_of(sid)
            st = self.states[k]
            prev = st.hbar.get(sid, 0.0)
            st.hbar[sid] = g * float(h) + (1 - g) * prev

        for k, st in self.states.items():
            for sid in list(st.r):
                if sid in sel_set:
                    st.r[sid] = 0.0
                else:
                    st.r[sid] += 1.0 + self.cfg.delta_in * max(0.0, st.hbar.get(sid, 0.0))

    def resync(self) -> None:
        self._sync_substates()

    def priorities(self) -> dict[int, float]:
        return {k: st.q for k, st in self.states.items()}
