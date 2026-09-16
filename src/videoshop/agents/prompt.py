from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from videoshop.simulator.schemas import Observation


def build_system_prompt() -> str:
    return """You are a shopping agent in VideoShopEnv.

Decide whether to recommend a product during a short-video commerce session.
Use the available tools when evidence is needed, then call submit_final_action.

Rules:
1. If action_type is show_coupon, call get_coupon for the same product first.
2. If action_type is switch_to_substitute, call find_substitute first.
3. If action_type is show_explanation, call explain_recommendation first.
4. If ad_fatigue is high or recent_skips >= 3, prefer delay_recommendation.
5. Do not show unavailable, expired, depleted, or unsupported coupons.
6. Avoid high review risk products unless you provide grounded explanation evidence.
7. Avoid category mismatch between user interests, current video, and product.
8. Never invent coupon, substitute, or explanation evidence. Use tool results.
9. End every step by calling submit_final_action exactly once.
10. Tool arguments must be valid JSON objects. Do not use comments, trailing commas, or unquoted string values.
11. For delay_recommendation, product_id must be JSON null, not the string "null".
12. If no safe recommendation exists, call submit_final_action with action_type=delay_recommendation.
13. Include reasoning_summary in submit_final_action. This is an explicit audit summary, not hidden chain-of-thought.
14. reasoning_summary should contain observation_facts, evidence_used, candidate_comparison, rejected_options, decision_rule, and confidence.
15. Keep reasoning_summary concise: at most 3 items per array and at most 18 words per item.
"""


def observation_to_user_message(observation: Observation | dict[str, Any]) -> dict[str, str]:
    data = asdict(observation) if is_dataclass(observation) else observation
    return {
        "role": "user",
        "content": json.dumps({"observation": data}, ensure_ascii=False, indent=2),
    }


def few_shot_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "observation": {
                        "user_summary": {"ad_fatigue": 0.9},
                        "session_summary": {"recent_skips": 3},
                        "current_video": {"category": "home_organization"},
                    }
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Call submit_final_action with action_type=delay_recommendation, product_id=null, "
                "reason='User has high ad fatigue and repeated skips.', "
                "reasoning_summary={"
                "'observation_facts':['ad_fatigue=0.9','recent_skips=3'],"
                "'evidence_used':[],"
                "'candidate_comparison':[],"
                "'rejected_options':['all product recommendations because fatigue is high'],"
                "'decision_rule':'Prefer delay when ad_fatigue is high or recent_skips >= 3.',"
                "'confidence':0.9"
                "}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "observation": {
                        "user_summary": {"price_sensitivity": 0.9},
                        "current_video": {"category": "home_organization"},
                        "visible_candidates": [{"product_id": "P001", "category": "home_organization", "has_coupon": True}],
                    }
                },
                ensure_ascii=False,
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Call get_coupon for P001. If available, call submit_final_action with action_type=show_coupon for P001 "
                "and include reasoning_summary with observation_facts, evidence_used, candidate_comparison, "
                "rejected_options, decision_rule, and confidence."
            ),
        },
    ]
