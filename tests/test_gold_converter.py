from __future__ import annotations

import json

from videoshop.training.gold_converter import (
    build_action_json_sft_samples,
    build_openai_tool_sft_samples,
    gold_rejection_reasons,
    select_gold_episodes,
)


def _episode(*, repaired: bool = False) -> dict:
    metadata = {"agent_repair_warnings": ["missing_final_action"]} if repaired else {}
    reasoning = {
        "observation_facts": ["coupon is available"],
        "evidence_used": ["get_coupon returned available=true"],
        "candidate_comparison": ["P001 matches the current video"],
        "rejected_options": [],
        "decision_rule": "Use only a verified coupon.",
        "confidence": 0.9,
    }
    return {
        "episode_id": "llm:scenario:001",
        "scenario_id": "scenario",
        "outcome": "purchase",
        "total_reward": 8.0,
        "metrics": {"constraint_violations": 0},
        "steps": [
            {
                "t": 1,
                "scenario_id": "scenario",
                "observation": {"visible_candidates": [{"product_id": "P001"}]},
                "state": {"candidate_products": [{"product_id": "P001"}]},
                "agent_step": {
                    "tool_requests": [{"tool": "get_coupon", "args": {"product_id": "P001"}}],
                    "final_action": {
                        "action_type": "show_coupon",
                        "product_id": "P001",
                        "reason": "A verified coupon is available.",
                        "evidence_refs": [],
                        "reasoning_summary": reasoning,
                    },
                    "metadata": metadata,
                },
                "reasoning_summary": reasoning,
                "tool_results": [
                    {
                        "tool": "get_coupon",
                        "input": {"product_id": "P001"},
                        "output": {"available": True},
                        "success": True,
                        "error": None,
                    }
                ],
                "reward": 8.0,
                "violations": [],
                "user_response": {
                    "clicked": True,
                    "added_to_cart": True,
                    "purchased": True,
                    "returned_or_refunded": False,
                },
            }
        ],
    }


def test_strict_gold_filter_rejects_repaired_episode():
    gold, rejected = select_gold_episodes([_episode(), _episode(repaired=True)])

    assert len(gold) == 1
    assert "agent_repair_warning" in next(iter(rejected.values()))


def test_gold_filter_requires_grounded_action_tool():
    episode = _episode()
    episode["steps"][0]["tool_results"] = []

    assert "missing_action_evidence" in gold_rejection_reasons(episode)


def test_openai_export_contains_tool_result_and_final_action_call():
    sample = build_openai_tool_sft_samples([_episode()])[0]
    messages = sample["messages"]

    assert sample["task"] == "videoshop_agent_tool_calling_sft"
    assert sample["metadata"]["tool_sequence_reconstructed"] is True
    assert messages[1]["role"] == "user"
    assert "observation" in json.loads(messages[1]["content"])
    assert messages[2]["tool_calls"][0]["function"]["name"] == "get_coupon"
    assert messages[3]["role"] == "tool"
    assert messages[4]["tool_calls"][0]["function"]["name"] == "submit_final_action"
    final_args = json.loads(messages[4]["tool_calls"][0]["function"]["arguments"])
    assert "evidence_refs" not in final_args


def test_action_json_export_preserves_reasoning_summary():
    sample = build_action_json_sft_samples([_episode()])[0]
    target = json.loads(sample["messages"][-1]["content"])

    assert target["final_action"]["reasoning_summary"]["confidence"] == 0.9
    assert "evidence_refs" not in target["final_action"]
    assert target["tool_requests"][0]["tool"] == "get_coupon"
