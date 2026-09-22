from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.data.scenarios import build_toy_scenarios
from videoshop.feed.legacy import (
    LEGACY_ACTION_TO_TREATMENT,
    exposure_from_legacy_step,
    is_feed_episode,
    legacy_action_of,
    treatment_for_legacy_action,
    upgrade_v1_episode,
)
from videoshop.policies import RuleBasedPolicy
from videoshop.simulator.benchmark import run_scenario_episode
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.rollout import run_episode
from videoshop.simulator.trajectory import SCHEMA_VERSION_V1, schema_version_of


@pytest.fixture()
def legacy_episode():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=4, seed=3)
    return run_episode(env, RuleBasedPolicy(), "E1")


def test_v1_writers_stamp_their_schema_version(legacy_episode):
    assert legacy_episode["schema_version"] == SCHEMA_VERSION_V1

    scenarios = build_toy_scenarios()
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), scenarios=scenarios, max_steps=4, seed=3)
    scenario_episode = run_scenario_episode(env, RuleBasedPolicy(), scenarios[0].scenario_id, "E1", seed=3)
    assert scenario_episode["schema_version"] == SCHEMA_VERSION_V1


def test_untagged_episodes_are_treated_as_v1():
    assert schema_version_of({"episode_id": "old"}) == SCHEMA_VERSION_V1
    assert not is_feed_episode({"episode_id": "old"})
    assert is_feed_episode({"schema_version": "v2"})


def test_every_legacy_action_maps_to_a_treatment():
    assert set(LEGACY_ACTION_TO_TREATMENT) == {
        "delay_recommendation",
        "show_product_card",
        "show_coupon",
        "switch_to_substitute",
        "show_explanation",
    }
    assert treatment_for_legacy_action("show_coupon") == "coupon"
    assert treatment_for_legacy_action("delay_recommendation") == "none"
    with pytest.raises(ValueError, match="Unknown legacy action_type"):
        treatment_for_legacy_action("serve_exposure")


def test_delay_step_becomes_an_organic_exposure():
    step = {
        "t": 1,
        "state": {"current_video": {"video_id": "V9"}},
        "action": {"action_type": "delay_recommendation", "product_id": None},
    }
    candidate = exposure_from_legacy_step(step)

    assert candidate.is_organic
    assert candidate.product_id is None
    assert candidate.treatment == "none"
    assert candidate.video_id == "V9"


def test_coupon_step_becomes_a_commercial_coupon_exposure():
    step = {
        "t": 2,
        "state": {"current_video": {"video_id": "V9"}},
        "action": {"action_type": "show_coupon", "product_id": "P1"},
    }
    candidate = exposure_from_legacy_step(step)

    assert candidate.is_commercial
    assert candidate.treatment == "coupon"
    assert candidate.product_id == "P1"


def test_upgrade_preserves_every_step_and_the_legacy_reward(legacy_episode):
    upgraded = upgrade_v1_episode(legacy_episode)

    assert upgraded["schema_version"] == "v2"
    assert upgraded["converted_from"] == SCHEMA_VERSION_V1
    assert len(upgraded["steps"]) == len(legacy_episode["steps"])
    assert upgraded["legacy_total_reward"] == legacy_episode["total_reward"]

    for original, converted in zip(legacy_episode["steps"], upgraded["steps"]):
        assert converted["legacy_reward"] == original["reward"]
        assert converted["user_response"] == original["user_response"]
        assert converted["decision"]["action_type"] == "serve_exposure"
        assert converted["served_exposure"]["exposure_id"] == converted["decision"]["exposure_id"]


def test_upgrade_does_not_invent_a_reward_vector(legacy_episode):
    upgraded = upgrade_v1_episode(legacy_episode)

    assert upgraded["total_reward_vector"] is None
    assert all(step["reward_vector"] is None for step in upgraded["steps"])


def test_upgrade_is_idempotent_for_v2_episodes():
    feed_episode = {"schema_version": "v2", "episode_id": "E1", "steps": []}
    assert upgrade_v1_episode(feed_episode) == feed_episode


def test_upgrade_rejects_unknown_versions():
    with pytest.raises(ValueError, match="Cannot upgrade unknown schema_version"):
        upgrade_v1_episode({"schema_version": "v7"})


def test_random_policy_episodes_also_convert():
    from videoshop.policies import RandomPolicy

    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=4, seed=3)
    episode = run_episode(env, RandomPolicy(random.Random(1)), "E2")
    upgraded = upgrade_v1_episode(episode)

    assert len(upgraded["steps"]) == len(episode["steps"])
    for step in upgraded["steps"]:
        assert step["served_exposure"]["source_type"] in {"organic", "seller"}


# --- LLM tool-calling layout (agent_step.final_action) ---------------------


def llm_step(action_type: str, product_id: str | None, video_id: str = "V001") -> dict:
    """A step shaped like scripts/generate_llm_trajectories.py writes it: no 'action' key."""
    return {
        "t": 1,
        "observation": {"current_video": {"video_id": video_id}},
        "state": {"current_video": {"video_id": video_id}},
        "agent_step": {
            "tool_requests": [],
            "final_action": {
                "action_type": action_type,
                "product_id": product_id,
                "reason": "because",
                "evidence_refs": [],
                "reasoning_summary": {"confidence": 0.8},
            },
            "metadata": {},
        },
        "tool_results": [
            {"tool": "get_coupon", "input": {"product_id": product_id}, "output": {"available": True}, "success": True},
            {"tool": "rank_products", "input": {}, "output": {}, "success": False},
        ],
        "reward": 2.0,
        "user_response": {"clicked": True},
    }


def test_llm_layout_action_is_found_without_an_action_key():
    step = llm_step("show_coupon", "P1")
    assert "action" not in step

    action = legacy_action_of(step)
    assert action["action_type"] == "show_coupon"
    assert action["product_id"] == "P1"


def test_llm_coupon_step_converts_to_a_commercial_coupon_exposure():
    candidate = exposure_from_legacy_step(llm_step("show_coupon", "P1"))

    assert candidate.source_type == "seller"
    assert candidate.treatment == "coupon"
    assert candidate.product_id == "P1"
    assert candidate.video_id == "V001"


@pytest.mark.parametrize(
    "action_type,expected_treatment",
    [
        ("show_product_card", "product_anchor"),
        ("show_coupon", "coupon"),
        ("switch_to_substitute", "product_anchor"),
        ("show_explanation", "product_anchor"),
    ],
)
def test_every_commercial_llm_action_keeps_its_product(action_type, expected_treatment):
    candidate = exposure_from_legacy_step(llm_step(action_type, "P7"))

    assert candidate.is_commercial
    assert candidate.treatment == expected_treatment
    assert candidate.product_id == "P7"


def test_llm_episode_upgrade_preserves_reason_and_tool_evidence():
    episode = {"episode_id": "E1", "steps": [llm_step("show_coupon", "P1")]}
    step = upgrade_v1_episode(episode)["steps"][0]

    assert step["decision"]["reason"] == "because"
    assert step["decision"]["reasoning_summary"] == {"confidence": 0.8}
    assert [call["tool"] for call in step["legacy_tool_calls"]] == ["get_coupon"], "failed tool calls are dropped"


def test_a_step_with_no_recoverable_action_raises_instead_of_defaulting():
    with pytest.raises(ValueError, match="carries no final action"):
        exposure_from_legacy_step({"t": 1, "state": {"current_video": {"video_id": "V1"}}})


def test_malformed_product_action_is_downgraded_but_recorded():
    warnings: list[str] = []
    candidate = exposure_from_legacy_step(llm_step("show_product_card", None), warnings=warnings)

    assert candidate.is_organic
    assert warnings == ["product_action_without_product_id:show_product_card"]


# --- regression against the trajectories already on disk -------------------

LLM_TRAJECTORY_FILES = sorted(
    path
    for path in (Path(__file__).resolve().parents[1] / "outputs" / "llm_trajectories").glob("*.jsonl")
)


@pytest.mark.parametrize("path", LLM_TRAJECTORY_FILES, ids=lambda path: path.name)
def test_recorded_llm_trajectories_survive_conversion(path):
    """Guards the real Qwen/DeepSeek runs: no commercial action may silently vanish."""
    recorded = Counter()
    converted = Counter()
    warnings = Counter()

    for line in path.read_text(encoding="utf-8").splitlines():
        episode = json.loads(line)
        if not episode.get("steps"):
            continue
        for step in episode["steps"]:
            recorded[step["agent_step"]["final_action"]["action_type"]] += 1

        upgraded = upgrade_v1_episode(episode)
        for step in upgraded["steps"]:
            served = step["served_exposure"]
            converted[f"{served['source_type']}/{served['treatment']}"] += 1
        warnings.update(upgraded["conversion_warnings"])

    assert recorded, f"{path.name} has no usable steps"

    expected_commercial = sum(
        count for action, count in recorded.items() if action != "delay_recommendation"
    ) - sum(warnings.values())
    actual_commercial = sum(count for key, count in converted.items() if not key.startswith("organic"))

    assert actual_commercial == expected_commercial
    assert converted["seller/coupon"] == recorded.get("show_coupon", 0)
    assert sum(converted.values()) == sum(recorded.values())


def test_recorded_llm_trajectories_are_actually_commercial():
    """A guard on the guard: if the fixtures held only delays, the test above would pass vacuously."""
    total_commercial = 0
    for path in LLM_TRAJECTORY_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            for step in episode.get("steps", []):
                action = step["agent_step"]["final_action"]["action_type"]
                total_commercial += action != "delay_recommendation"

    assert LLM_TRAJECTORY_FILES, "no recorded LLM trajectories found"
    assert total_commercial > 100
