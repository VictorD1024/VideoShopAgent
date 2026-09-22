from __future__ import annotations

import copy
import random
from dataclasses import asdict

from videoshop.data.synthetic import (
    SyntheticCommerceConfig,
    build_synthetic_catalog,
    build_synthetic_users,
    build_synthetic_videos,
)
from videoshop.feed.candidate_provider import CandidateBatch, CandidateProviderConfig, SyntheticCandidateProvider
from videoshop.feed.reward import RewardConfig, expected_reward_vector
from videoshop.feed.schemas import ScenarioSpec
from videoshop.simulator.schemas import SessionState, UserProfile

__all__ = [
    "NEUTRAL_INSTRUCTIONS",
    "OBJECTIVE_PROFILES",
    "SESSION_FAMILIES",
    "build_default_provider",
    "build_feed_scenario",
    "build_smoke_scenarios",
]

# Deliberately neutral: they describe the job, never the answer.
NEUTRAL_INSTRUCTIONS = (
    "Choose what this viewer sees next in the feed.",
    "Select the next exposure for this session.",
    "Decide which candidate to serve at this point in the session.",
    "Pick the next item for this viewer from the candidate set.",
)

OBJECTIVE_PROFILES = ("balanced", "content_first", "gmv_first", "retention_first", "clearance_campaign")

# Session archetypes. Stored only in the private oracle so the agent cannot
# pattern-match on a scenario type label.
SESSION_FAMILIES = (
    "fresh_browser",
    "fatigued_viewer",
    "high_intent_shopper",
    "price_sensitive",
    "risk_averse",
    "deep_session",
)


def build_smoke_scenarios(
    count: int = 100,
    seed: int = 7,
    *,
    provider_config: CandidateProviderConfig | None = None,
    commerce_config: SyntheticCommerceConfig | None = None,
    scenario_prefix: str = "feed_smoke",
    provider: SyntheticCandidateProvider | None = None,
    users: list[UserProfile] | None = None,
) -> list[ScenarioSpec]:
    """Small leak-free scenario set for validating the v0.2 feed protocol.

    Pass ``provider`` and ``users`` to draw from a split's private entity pool;
    the defaults build a standalone world for ad-hoc use.
    """

    commerce_config = commerce_config or SyntheticCommerceConfig(seed=seed)
    provider = provider or build_default_provider(provider_config, commerce_config)
    users = users or build_synthetic_users(commerce_config)
    rng = random.Random(seed)

    scenarios: list[ScenarioSpec] = []
    for index in range(count):
        user = copy.deepcopy(rng.choice(users))
        family = SESSION_FAMILIES[index % len(SESSION_FAMILIES)]
        session = _apply_family(user, family, rng)
        scenarios.append(
            build_feed_scenario(
                scenario_id=f"{scenario_prefix}_{index + 1:04d}",
                provider=provider,
                user=user,
                session=session,
                rng=rng,
                reward_config=RewardConfig(profile=OBJECTIVE_PROFILES[index % len(OBJECTIVE_PROFILES)]),
                instruction=NEUTRAL_INSTRUCTIONS[index % len(NEUTRAL_INSTRUCTIONS)],
                family=family,
            )
        )
    return scenarios


def build_default_provider(
    provider_config: CandidateProviderConfig | None = None,
    commerce_config: SyntheticCommerceConfig | None = None,
) -> SyntheticCandidateProvider:
    commerce_config = commerce_config or SyntheticCommerceConfig()
    return SyntheticCandidateProvider(
        build_synthetic_catalog(commerce_config),
        build_synthetic_videos(commerce_config),
        provider_config,
    )


def build_feed_scenario(
    *,
    scenario_id: str,
    provider: SyntheticCandidateProvider,
    user: UserProfile,
    session: SessionState,
    rng: random.Random,
    reward_config: RewardConfig | None = None,
    instruction: str = NEUTRAL_INSTRUCTIONS[0],
    family: str = "unspecified",
    placement: str = "for_you",
) -> ScenarioSpec:
    reward_config = reward_config or RewardConfig()
    batch = provider.generate(
        user,
        session,
        session.step,
        rng,
        exposure_prefix=f"{scenario_id}:",
        placement=placement,
    )
    return ScenarioSpec(
        scenario_id=scenario_id,
        instruction=instruction,
        public_context={
            "placement": placement,
            "user_summary": _redacted_user_summary(user),
            "session_summary": asdict(session),
            "catalog": batch.public_catalog(),
        },
        hidden_world_state={
            "user_profile": asdict(user),
            "session_state": asdict(session),
            "exposure_truth": {key: value.to_dict() for key, value in batch.truth.items()},
            "videos": {key: asdict(value) for key, value in batch.videos.items()},
            "products": {key: asdict(value) for key, value in batch.products.items()},
        },
        candidate_exposures=list(batch.candidates),
        oracle=_build_oracle(batch, user, session, reward_config, family),
        reward_config=reward_config,
    )


def _build_oracle(
    batch: CandidateBatch,
    user: UserProfile,
    session: SessionState,
    reward_config: RewardConfig,
    family: str,
) -> dict:
    """Evaluator-private reference. Greedy over the full-information expectation."""

    scores: dict[str, float] = {}
    for candidate in batch.candidates:
        if not candidate.eligibility.is_eligible:
            continue
        vector = expected_reward_vector(candidate, batch.truth[candidate.exposure_id], user, session.recent_skips)
        scores[candidate.exposure_id] = round(reward_config.scalarize(vector), 6)

    gold = max(scores, key=lambda key: scores[key]) if scores else None
    return {
        "session_family": family,
        "gold_exposure_id": gold,
        "expected_scalar_by_exposure": scores,
        "gold_scalar": scores.get(gold) if gold else None,
        "worst_eligible_scalar": min(scores.values()) if scores else None,
    }


def _apply_family(user: UserProfile, family: str, rng: random.Random) -> SessionState:
    session = SessionState()

    if family == "fresh_browser":
        user.ad_fatigue = round(rng.uniform(0.02, 0.15), 2)
        user.purchase_intent = round(rng.uniform(0.05, 0.25), 2)
    elif family == "fatigued_viewer":
        user.ad_fatigue = round(rng.uniform(0.70, 0.92), 2)
        session.recent_skips = rng.randint(3, 5)
    elif family == "high_intent_shopper":
        user.purchase_intent = round(rng.uniform(0.70, 0.92), 2)
        user.ad_fatigue = round(rng.uniform(0.10, 0.35), 2)
    elif family == "price_sensitive":
        user.price_sensitivity = round(rng.uniform(0.80, 0.97), 2)
        user.purchase_intent = round(rng.uniform(0.35, 0.65), 2)
    elif family == "risk_averse":
        user.risk_sensitivity = round(rng.uniform(0.80, 0.95), 2)
        user.purchase_intent = round(rng.uniform(0.25, 0.55), 2)
    elif family == "deep_session":
        session.step = 0
        session.recent_skips = rng.randint(1, 2)
        session.recent_watch_categories = [
            rng.choice(list(user.category_interests)) for _ in range(rng.randint(2, 4))
        ]
        user.ad_fatigue = round(rng.uniform(0.35, 0.60), 2)

    return session


def _redacted_user_summary(user: UserProfile) -> dict:
    payload = asdict(user)
    payload.pop("purchase_intent", None)  # a response parameter, not a public feature
    payload["ad_fatigue"] = round(payload["ad_fatigue"], 4)
    return payload
