from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from videoshop.data.synthetic import SyntheticCommerceConfig
from videoshop.feed.candidate_provider import CandidateProviderConfig, SyntheticCandidateProvider
from videoshop.feed.composite_env import FeedInterventionEnv
from videoshop.feed.control_env import FeedControlEnv
from videoshop.feed.policies import build_feed_policy, build_intervention_policy
from videoshop.feed.reward import REWARD_COMPONENTS
from videoshop.feed.schemas import FEED_SCHEMA_VERSION, ScenarioSpec
from videoshop.feed.scenarios import build_smoke_scenarios
from videoshop.feed.splits import (
    SPLIT_ORDER,
    SPLIT_SEEDS,
    SPLIT_SIZES,
    EntityPool,
    build_pool_provider,
    partition_world,
)

FEED_SPLITS = SPLIT_ORDER


@dataclass(frozen=True)
class FeedSplit:
    """A split's scenarios together with the provider that owns its entities.

    Both are returned as a unit because they must agree: the provider that produced
    a scenario's step-0 candidates is also the one that must produce its later steps.
    """

    name: str
    scenarios: list[ScenarioSpec]
    provider: SyntheticCandidateProvider
    pool: EntityPool

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)


def build_split(
    split: str,
    *,
    count: int | None = None,
    provider_config: CandidateProviderConfig | None = None,
    world_config: SyntheticCommerceConfig | None = None,
) -> FeedSplit:
    if split not in FEED_SPLITS:
        raise ValueError(f"Unsupported feed split: {split}. Available: {list(FEED_SPLITS)}")

    pool = partition_world(world_config)[split]
    provider = build_pool_provider(pool, provider_config)
    scenarios = build_smoke_scenarios(
        count=count if count is not None else SPLIT_SIZES[split],
        seed=SPLIT_SEEDS[split],
        scenario_prefix=f"feed_{split}",
        provider=provider,
        users=pool.users,
    )
    return FeedSplit(name=split, scenarios=scenarios, provider=provider, pool=pool)


def dataset_hash(scenarios: list[ScenarioSpec]) -> str:
    digest = hashlib.sha256()
    for scenario in scenarios:
        digest.update(json.dumps(scenario.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()[:16]


class OracleStepGreedyPolicy:
    """Evaluator-only ceiling reference.

    It reads the environment's private truth directly, so it is not a legal agent;
    it exists to show how much headroom a benchmark number still has.
    """

    name = "oracle_step_greedy"

    def __init__(self, env: FeedControlEnv) -> None:
        self.env = env

    def act(self, observation) -> Any:
        from videoshop.feed.reward import expected_reward_vector
        from videoshop.feed.schemas import FeedDecision

        del observation
        env = self.env
        if env.batch is None or env.scenario is None or env.user is None or env.session is None:
            raise RuntimeError("OracleStepGreedyPolicy requires a reset environment.")

        best_id, best_score = None, float("-inf")
        for candidate in env.batch.candidates:
            if not candidate.eligibility.is_eligible:
                continue
            vector = expected_reward_vector(
                candidate, env.batch.truth[candidate.exposure_id], env.user, env.session.recent_skips
            )
            score = env.scenario.reward_config.scalarize(vector)
            if score > best_score:
                best_id, best_score = candidate.exposure_id, score

        target = best_id or env.batch.candidates[0].exposure_id
        return FeedDecision.serve(target, reason="Oracle step-greedy reference.")


def run_feed_episode(
    env: FeedControlEnv,
    policy,
    scenario: ScenarioSpec,
    episode_id: str,
    *,
    seed: int,
) -> dict[str, Any]:
    observation = env.reset(scenario, seed=seed)
    done = False
    while not done:
        observation, _vector, terminated, truncated, _info = env.step(policy.act(observation))
        done = terminated or truncated

    episode = env.trajectory(episode_id=episode_id, policy=getattr(policy, "name", "unknown"))
    episode["terminated_by"] = "session_exit" if terminated else "max_steps"
    episode["oracle"] = {
        "gold_exposure_id": scenario.oracle.get("gold_exposure_id"),
        "gold_scalar": scenario.oracle.get("gold_scalar"),
        "session_family": scenario.oracle.get("session_family"),
    }
    return episode


def run_feed_benchmark(
    policy_name: str,
    *,
    split: str | FeedSplit = "smoke",
    scenarios: list[ScenarioSpec] | None = None,
    provider: SyntheticCandidateProvider | None = None,
    seed: int = 42,
    max_steps: int = 6,
    episodes_per_scenario: int = 1,
    count: int | None = None,
    provider_config: CandidateProviderConfig | None = None,
    intervention: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one policy over a split.

    Pass a :class:`FeedSplit`, or a split name, or both ``scenarios`` and ``provider``.
    Supplying scenarios without their provider used to fall back to the default split's
    provider, which silently ran a frozen_eval scenario against the smoke catalog.
    """

    if isinstance(split, FeedSplit):
        if scenarios is not None or provider is not None:
            raise ValueError("Pass a FeedSplit or explicit scenarios/provider, not both.")
        scenarios, provider, split = split.scenarios, split.provider, split.name
    elif (scenarios is None) != (provider is None):
        raise ValueError(
            "scenarios and provider must be supplied together; a scenario set is only "
            "meaningful alongside the provider that owns its entities."
        )
    elif scenarios is None:
        built = build_split(split, count=count, provider_config=provider_config)
        scenarios, provider = built.scenarios, built.provider
    env = (
        FeedInterventionEnv(
            provider, build_intervention_policy(intervention), max_steps=max_steps, seed=seed
        )
        if intervention
        else FeedControlEnv(provider, max_steps=max_steps, seed=seed)
    )

    episodes: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        for repeat in range(episodes_per_scenario):
            episode_seed = seed + index * episodes_per_scenario + repeat
            policy = (
                OracleStepGreedyPolicy(env)
                if policy_name == OracleStepGreedyPolicy.name
                else build_feed_policy(policy_name, seed=episode_seed)
            )
            episodes.append(
                run_feed_episode(
                    env,
                    policy,
                    scenario,
                    f"{scenario.scenario_id}:{repeat + 1:03d}",
                    seed=episode_seed,
                )
            )

    summary = summarize_feed_benchmark(episodes, policy_name=policy_name)
    summary["split"] = split
    summary["schema_version"] = FEED_SCHEMA_VERSION
    summary["seed"] = seed
    summary["max_steps"] = max_steps
    summary["dataset_hash"] = dataset_hash(scenarios)
    summary["intervention"] = intervention
    summary["provider_config"] = asdict(provider.config)
    summary["entity_pool"] = {
        "products": len(provider.products),
        "videos": len(provider.videos),
    }
    return episodes, summary


def summarize_feed_benchmark(episodes: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    if not episodes:
        return {"policy": policy_name, "episodes": 0}

    totals = [episode["total_scalar_reward"] for episode in episodes]
    steps = [step for episode in episodes for step in episode["steps"]]
    served = [step for step in steps if step["served_exposure"]]
    responses = [step["user_response"] for step in served if step["user_response"]]

    commercial = sum(1 for step in served if step["served_exposure"]["source_type"] != "organic")
    by_source: dict[str, int] = {}
    for step in served:
        source = step["served_exposure"]["source_type"]
        by_source[source] = by_source.get(source, 0) + 1

    gold_matches = sum(
        1
        for episode in episodes
        if episode["steps"] and episode["steps"][0]["decision"]["exposure_id"] == episode["oracle"]["gold_exposure_id"]
    )

    by_profile: dict[str, list[float]] = {}
    for episode in episodes:
        by_profile.setdefault(episode["objective_profile"], []).append(episode["total_scalar_reward"])

    low, high = bootstrap_ci(totals)
    return {
        "policy": policy_name,
        "episodes": len(episodes),
        "scenarios": len({episode["scenario_id"] for episode in episodes}),
        "avg_scalar_reward": round(mean(totals), 4),
        "scalar_reward_ci95": [low, high],
        "avg_scalar_reward_by_profile": {
            profile: round(mean(values), 4) for profile, values in sorted(by_profile.items())
        },
        "avg_reward_vector": {
            component: round(
                mean(episode["total_reward_vector"][component] for episode in episodes), 4
            )
            for component in REWARD_COMPONENTS
        },
        "avg_session_length": round(mean(len(episode["steps"]) for episode in episodes), 3),
        "session_exit_rate": round(
            sum(episode["terminated_by"] == "session_exit" for episode in episodes) / len(episodes), 4
        ),
        "blocked_decision_rate": round(sum(step["blocked"] for step in steps) / max(len(steps), 1), 4),
        "commercial_exposure_share": round(commercial / max(len(served), 1), 4),
        "served_by_source": by_source,
        "skip_rate": _rate(responses, "skipped"),
        "click_rate": _rate(responses, "clicked"),
        "purchase_rate": _rate(responses, "purchased"),
        "refund_rate": _rate(responses, "refunded"),
        "annoyance_rate": _rate(responses, "annoyed"),
        "first_step_gold_match_rate": round(gold_matches / len(episodes), 4),
    }


def bootstrap_ci(values: list[float], samples: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) < 2:
        return (round(values[0], 4), round(values[0], 4)) if values else (0.0, 0.0)
    rng = random.Random(seed)
    size = len(values)
    means = sorted(mean(rng.choice(values) for _ in range(size)) for _ in range(samples))
    low = means[int(samples * alpha / 2)]
    high = means[min(samples - 1, int(samples * (1 - alpha / 2)))]
    return round(low, 4), round(high, 4)


def _rate(responses: list[dict[str, Any]], key: str) -> float:
    if not responses:
        return 0.0
    return round(sum(bool(response.get(key)) for response in responses) / len(responses), 4)
