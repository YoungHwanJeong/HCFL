# Maps NVFlare site names (site-1..site-N) to HCFL parent ids (0..N-1).
from __future__ import annotations


def site_name(parent: int) -> str:
    return f"site-{parent + 1}"


def parent_id(name: str) -> int:
    return int(name.split("-")[-1]) - 1
