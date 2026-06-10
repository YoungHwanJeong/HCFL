# Simulated per-client resource profiles and local-training runtime.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResourceProfile:
    cpu_cores: int
    memory_gb: float
    gpu_frac: float
    speed_jitter: float

    @property
    def throughput(self) -> float:
        return 8.0 * self.gpu_frac + 0.05 * self.cpu_cores


def sample_profiles(num_clients: int, cfg, seed: int = 42) -> list[ResourceProfile]:
    rng = np.random.default_rng(seed)
    lo_c, hi_c = cfg.cpu_cores_range
    lo_m, hi_m = cfg.memory_gb_range
    lo_g, hi_g = cfg.gpu_frac_range
    profiles = []
    for _ in range(num_clients):
        profiles.append(
            ResourceProfile(
                cpu_cores=int(rng.integers(lo_c, hi_c + 1)),
                memory_gb=float(rng.uniform(lo_m, hi_m)),
                gpu_frac=float(rng.uniform(lo_g, hi_g)),
                speed_jitter=float(rng.uniform(0.85, 1.15)),
            )
        )
    return profiles


_REF_THROUGHPUT = 8.0 * 0.15 + 0.05 * 9.0
_REF_WORKLOAD = 300 * 5


def estimate_runtime(
    profile: ResourceProfile,
    n_samples: int,
    local_epochs: int,
    *,
    rng: np.random.Generator | None = None,
) -> float:
    workload = max(n_samples, 1) * local_epochs
    base = (workload / _REF_WORKLOAD) * (_REF_THROUGHPUT / max(profile.throughput, 1e-6))
    noise = profile.speed_jitter
    if rng is not None:
        noise *= float(rng.uniform(0.9, 1.1))
    return float(base * noise)
