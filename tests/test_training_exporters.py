from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.policies import RuleBasedPolicy
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.rollout import run_episode
from videoshop.training.exporters import build_hard_negative, export_dpo_pairs, export_rl_rollouts, export_sft_samples


def _episode() -> dict:
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=3, seed=42)
    return run_episode(env, RuleBasedPolicy(), "E000001")


def test_export_sft_samples_uses_tool_calls_and_action_as_target():
    samples = export_sft_samples([_episode()])

    first = samples[0]

    assert first["task"] == "videoshop_agent_action_sft"
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][2]["role"] == "assistant"
    assert "show_product_card" in first["messages"][2]["content"]


def test_export_dpo_pairs_contains_rule_based_hard_negative_metadata():
    pairs = export_dpo_pairs([_episode()])

    first = pairs[0]

    assert first["task"] == "videoshop_agent_action_dpo"
    assert first["metadata"]["preference_source"] == "rule_based_hard_negative"
    assert first["chosen"] != first["rejected"]
    assert first["metadata"]["negative_type"] in {
        "category_mismatch",
        "fake_coupon",
        "high_risk_unexplained",
        "missed_substitute",
        "premature_intervention",
        "unsupported_explanation",
        "weak_evidence",
    }


def test_build_hard_negative_for_delay_recommendation_is_premature_intervention():
    step = _episode()["steps"][0]
    step["action"] = {
        "action_type": "delay_recommendation",
        "product_id": None,
        "reason": "High ad fatigue or repeated skips.",
        "tool_calls": [],
        "evidence": {},
    }

    rejected, negative_type = build_hard_negative(step)

    assert negative_type == "premature_intervention"
    assert rejected["action_type"] == "show_product_card"
    assert rejected["product_id"] is not None


def test_build_hard_negative_for_price_sensitive_state_prefers_fake_coupon():
    step = _episode()["steps"][0]
    step["state"]["user_profile"]["price_sensitivity"] = 0.9

    rejected, negative_type = build_hard_negative(step)

    assert negative_type == "fake_coupon"
    assert rejected["action_type"] == "show_coupon"
    assert rejected["evidence"]["coupon"]["available"] is False


def test_build_hard_negative_for_substitute_action_is_missed_substitute():
    step = _episode()["steps"][0]
    step["action"] = {
        "action_type": "switch_to_substitute",
        "product_id": "P006",
        "reason": "Use a lower-price substitute.",
        "tool_calls": [
            {
                "tool": "find_substitute",
                "input": {"product_id": "P003", "constraints": {"lower_price": True}},
                "output": {"product_id": "P006"},
            }
        ],
        "evidence": {"substitute": {"product_id": "P006"}},
    }

    rejected, negative_type = build_hard_negative(step)

    assert negative_type == "missed_substitute"
    assert rejected["action_type"] == "show_product_card"
    assert rejected["product_id"] == "P003"


def test_export_rl_rollouts_has_next_state_and_terminal_done_flag():
    rollout = export_rl_rollouts([_episode()])[0]
    transitions = rollout["trajectory"]

    assert rollout["task"] == "videoshop_agent_rl_rollout"
    assert transitions[0]["next_state"] == transitions[1]["state"]
    assert transitions[-1]["done"] is True
