# Numpy array <-> bytes for moving arrays through NVFlare Shareables.
from __future__ import annotations

import io

import numpy as np


def ndarray_to_bytes(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)
