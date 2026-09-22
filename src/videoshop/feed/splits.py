from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from videoshop.data.synthetic import (
    SyntheticCommerceConfig,
    build_synthetic_catalog,
    build_synthetic_users,
    build_synthetic_videos,
)
from videoshop.feed.candidate_provider import CandidateProviderConfig, SyntheticCandidateProvider
from videoshop.simulator.schemas import Product, UserProfile, VideoContext

FEED_WORLD_CONFIG = SyntheticCommerceConfig(
    seed=20260918,
    category_count=12,
    products_per_category=90,
    users=150,
    videos_per_category=25,
)
"""The single world all feed splits are carved out of.

Splits must partition *entities*, not just random seeds. Re-seeding the generator
reproduces the same ``P0010018``-style ids with different attributes, which looks
disjoint by scenario id while the underlying catalog is fully shared.
"""

# Ordered so the partition is stable: appending a split must not reshuffle the others.
SPLIT_ORDER = ("train", "smoke", "frozen_eval")
SPLIT_SHARES = {"train": 0.6, "smoke": 0.2, "frozen_eval": 0.2}
SPLIT_SEEDS = {"train": 101, "smoke": 7, "frozen_eval": 9001}
SPLIT_SIZES = {"train": 500, "smoke": 100, "frozen_eval": 200}


@dataclass(frozen=True)
class EntityPool:
    """A split's private slice of the world. No entity id appears in two pools."""

    name: str
    products: list[Product] = field(default_factory=list)
    videos: list[VideoContext] = field(default_factory=list)
    users: list[UserProfile] = field(default_factory=list)

    def product_ids(self) -> set[str]:
        return {product.product_id for product in self.products}

    def video_ids(self) -> set[str]:
        return {video.video_id for video in self.videos}

    def user_ids(self) -> set[str]:
        return {user.user_id for user in self.users}

    def categories(self) -> set[str]:
        return {video.category for video in self.videos}


def partition_world(
    config: SyntheticCommerceConfig | None = None,
    shares: dict[str, float] | None = None,
) -> dict[str, EntityPool]:
    """Split one generated world into disjoint per-split entity pools.

    Products and videos are partitioned *within* each category so every split still
    covers every category; only the concrete items differ. Users are partitioned
    globally.
    """

    config = config or FEED_WORLD_CONFIG
    shares = shares or SPLIT_SHARES
    _validate_shares(shares)

    products_by_category = _group(build_synthetic_catalog(config), lambda item: item.category)
    videos_by_category = _group(build_synthetic_videos(config), lambda item: item.category)
    users = sorted(build_synthetic_users(config), key=lambda user: user.user_id)

    product_slices: dict[str, list[Product]] = {name: [] for name in shares}
    video_slices: dict[str, list[VideoContext]] = {name: [] for name in shares}

    for category in sorted(products_by_category):
        items = sorted(products_by_category[category], key=lambda item: item.product_id)
        for name, chunk in _slice_by_share(items, shares).items():
            product_slices[name].extend(chunk)
    for category in sorted(videos_by_category):
        items = sorted(videos_by_category[category], key=lambda item: item.video_id)
        for name, chunk in _slice_by_share(items, shares).items():
            video_slices[name].extend(chunk)
    user_slices = _slice_by_share(users, shares)

    result = {
        name: EntityPool(
            name=name,
            products=copy.deepcopy(product_slices[name]),
            videos=copy.deepcopy(video_slices[name]),
            users=copy.deepcopy(user_slices[name]),
        )
        for name in shares
    }

    for name, pool in result.items():
        if not pool.products or not pool.videos or not pool.users:
            raise ValueError(f"Split {name} received an empty entity pool; widen the world or the share.")
    return result


def build_pool_provider(
    pool: EntityPool,
    provider_config: CandidateProviderConfig | None = None,
) -> SyntheticCandidateProvider:
    """A provider restricted to one split's entities.

    The same provider must serve step 0 and every later step, otherwise an episode
    mixes two catalogs that share product ids but not attributes.
    """

    return SyntheticCandidateProvider(pool.products, pool.videos, provider_config)


def _slice_by_share(items: Sequence, shares: dict[str, float]) -> dict[str, list]:
    total = len(items)
    chunks: dict[str, list] = {}
    start = 0
    ordered = [name for name in SPLIT_ORDER if name in shares]
    ordered.extend(sorted(name for name in shares if name not in SPLIT_ORDER))

    for index, name in enumerate(ordered):
        if index == len(ordered) - 1:
            end = total
        else:
            end = start + int(round(total * shares[name]))
            end = min(end, total)
        chunks[name] = list(items[start:end])
        start = end
    return chunks


def _group(items: Iterable, key) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(key(item), []).append(item)
    return grouped


def _validate_shares(shares: dict[str, float]) -> None:
    if not shares:
        raise ValueError("At least one split share is required.")
    if any(value <= 0 for value in shares.values()):
        raise ValueError("Split shares must be positive.")
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split shares must sum to 1.0, got {total}.")
