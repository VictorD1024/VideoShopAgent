from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION_V1 = "v1"
"""Frozen schema for intervention-layer trajectories (VideoShopEnv).

v1 records a fixed current_video plus one final action per step. Feed-control
trajectories use v2 (see videoshop.feed.schemas.FEED_SCHEMA_VERSION); the two are
not interchangeable, so every writer stamps its version explicitly.
"""


def schema_version_of(episode: dict) -> str:
    """Untagged episodes predate versioning and are v1 by definition."""
    return episode.get("schema_version", SCHEMA_VERSION_V1)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

