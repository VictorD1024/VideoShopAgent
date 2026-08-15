from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ACTION_SPACE = [
    "delay_recommendation",
    "show_product_card(product_id)",
    "show_coupon(product_id)",
    "switch_to_substitute(product_id)",
    "show_explanation(product_id)",
]


TOOL_SPACE = [
    "retrieve_candidates(video_context, user_summary)",
    "rank_products(candidates, state)",
    "get_coupon(product_id, user_id)",
    "find_substitute(product_id, constraints)",
    "explain_recommendation(product_id, evidence)",
]


def export_sft_samples(episodes: list[dict]) -> list[dict]:
    samples = []
    for episode in episodes:
        for step in episode["steps"]:
            samples.append(
                {
                    "id": _sample_id(episode, step),
                    "task": "videoshop_agent_action_sft",
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": _state_prompt(step["state"])},
                        {"role": "assistant", "content": _json_dumps(_sft_target(step))},
                    ],
                    "metadata": _step_metadata(episode, step),
                }
            )
    return samples


def export_dpo_pairs(episodes: list[dict]) -> list[dict]:
    pairs = []
    for episode in episodes:
        for step in episode["steps"]:
            rejected_action, negative_type = build_hard_negative(step)
            pairs.append(
                {
                    "id": _sample_id(episode, step),
                    "task": "videoshop_agent_action_dpo",
                    "prompt": [
                        {"role": "system", "content": _system_prompt()},
                        {"role": "user", "content": _state_prompt(step["state"])},
                    ],
                    "chosen": _json_dumps(_sft_target(step)),
                    "rejected": _json_dumps(
                        {
                            "tool_calls": rejected_action.get("tool_calls", []),
                            "action": rejected_action,
                        }
                    ),
                    "metadata": {
                        **_step_metadata(episode, step),
                        "negative_type": negative_type,
                        "preference_source": "rule_based_hard_negative",
                    },
                }
            )
    return pairs


def export_rl_rollouts(episodes: list[dict]) -> list[dict]:
    rollouts = []
    for episode in episodes:
        steps = episode["steps"]
        transitions = []
        for idx, step in enumerate(steps):
            next_state = steps[idx + 1]["state"] if idx + 1 < len(steps) else None
            transitions.append(
                {
                    "t": step["t"],
                    "state": step["state"],
                    "tool_calls": step["tool_calls"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "user_response": step["user_response"],
                    "state_update": step["state_update"],
                    "next_state": next_state,
                    "done": idx == len(steps) - 1,
                }
            )

        rollouts.append(
            {
                "episode_id": episode["episode_id"],
                "task": "videoshop_agent_rl_rollout",
                "trajectory": transitions,
                "total_reward": episode["total_reward"],
                "outcome": episode["outcome"],
            }
        )
    return rollouts


def build_hard_negative(step: dict) -> tuple[dict, str]:
    state = step["state"]
    chosen = step["action"]
    products = state.get("candidate_products", [])
    session = state.get("session_state", {})
    user = state.get("user_profile", {})

    if chosen["action_type"] == "delay_recommendation":
        product = _best_product(state) or _first_product(products)
        return _negative_product_card(product, "Pushes a product despite high ad fatigue or repeated skips."), "premature_intervention"

    missed_substitute = _missed_substitute_product(step, products)
    if missed_substitute:
        return _negative_product_card(missed_substitute, "Ignores the lower-price or lower-risk substitute found by the tool."), "missed_substitute"

    chosen_product = _find_product(products, chosen.get("product_id"))
    if chosen_product and chosen_product.get("review_risk", 0.0) >= _max_review_risk(state):
        return _negative_product_card(chosen_product, "Ignores high review risk without grounded evidence."), "high_risk_unexplained"

    if _needs_price_help(user) and chosen_product:
        invalid_coupon_product = _product_without_available_coupon(state, products, preferred_category=chosen_product.get("category"))
        if invalid_coupon_product:
            return _negative_coupon(state, invalid_coupon_product), "fake_coupon"

    mismatched_product = _category_mismatch_product(state, products)
    if mismatched_product and step["t"] % 3 == 0:
        return _negative_product_card(mismatched_product, "Looks plausible but mismatches the current video category."), "category_mismatch"

    risky_match = _risky_matched_product(state, products, exclude_product_id=chosen.get("product_id"))
    if risky_match and not _needs_price_help(user):
        return _negative_product_card(risky_match, "Recommends a highly relevant product but ignores high review risk."), "high_risk_unexplained"

    if mismatched_product:
        return _negative_product_card(mismatched_product, "Looks plausible but mismatches the current video category."), "category_mismatch"

    if chosen["action_type"] != "show_explanation" and chosen_product:
        unsupported = deepcopy(chosen)
        unsupported["action_type"] = "show_explanation"
        unsupported["reason"] = "Explains the recommendation without tool-grounded evidence."
        unsupported["evidence"] = {}
        unsupported["tool_calls"] = []
        return unsupported, "unsupported_explanation"

    product = _first_product(products)
    return _negative_product_card(product, "Fallback hard negative with weaker evidence."), "weak_evidence"


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps(row) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sft_target(step: dict) -> dict:
    return {
        "tool_calls": step["tool_calls"],
        "action": step["action"],
    }


def _system_prompt() -> str:
    return (
        "You are VideoShopAgent, a short-video commerce decision agent. "
        "Use only the closed tool/action space, ground recommendations in video, user, product, and commerce evidence, "
        "and optimize conversion while avoiding fake coupons, high return risk, and user interruption."
    )


def _state_prompt(state: dict) -> str:
    compact_state = {
        "user_profile": state.get("user_profile", {}),
        "session_state": state.get("session_state", {}),
        "current_video": state.get("current_video", {}),
        "candidate_products": state.get("candidate_products", []),
        "commerce_context": state.get("commerce_context", {}),
        "available_tools": TOOL_SPACE,
        "available_actions": ACTION_SPACE,
    }
    return "Choose the next tool calls and final action for this state:\n" + _json_dumps(compact_state)


def _step_metadata(episode: dict, step: dict) -> dict:
    response = step["user_response"]
    return {
        "episode_id": episode["episode_id"],
        "t": step["t"],
        "reward": step["reward"],
        "outcome": episode["outcome"],
        "clicked": response.get("clicked", False),
        "added_to_cart": response.get("added_to_cart", False),
        "purchased": response.get("purchased", False),
        "interrupted": response.get("interrupted", False),
    }


def _sample_id(episode: dict, step: dict) -> str:
    return f"{episode['episode_id']}_T{step['t']:03d}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _product_match_score(state: dict, product: dict) -> float:
    user = state.get("user_profile", {})
    video = state.get("current_video", {})
    score = 0.0
    score += 0.40 * user.get("category_interests", {}).get(product.get("category"), 0.0)
    if product.get("category") == video.get("category"):
        score += 0.25
    if any(tag in user.get("style_preferences", []) for tag in product.get("tags", [])):
        score += 0.15
    title = product.get("title", "").lower()
    if any(obj in title for obj in video.get("objects", [])):
        score += 0.10
    if product.get("rating", 0.0) >= 4.5:
        score += 0.10
    return min(score, 1.0)


def _best_product(state: dict) -> dict | None:
    products = state.get("candidate_products", [])
    return max(products, key=lambda product: _product_match_score(state, product), default=None)


def _first_product(products: list[dict]) -> dict:
    if not products:
        return {"product_id": None, "title": "", "category": "", "review_risk": 0.0}
    return products[0]


def _find_product(products: list[dict], product_id: str | None) -> dict | None:
    return next((product for product in products if product.get("product_id") == product_id), None)


def _needs_price_help(user: dict) -> bool:
    return user.get("price_sensitivity", 0.0) > 0.65 or user.get("budget_level") == "low"


def _max_review_risk(state: dict) -> float:
    return state.get("commerce_context", {}).get("risk_constraints", {}).get("max_review_risk", 0.35)


def _coupon_available(state: dict, product: dict) -> bool:
    context = state.get("commerce_context", {})
    product_id = product.get("product_id")
    inventory = context.get("coupon_inventory", {}).get(product_id, 0)
    threshold = context.get("coupon_thresholds", {}).get(product_id, 0.0)
    expiry_step = context.get("coupon_expiry_steps", {}).get(product_id)
    campaign_budget = context.get("campaign_budget", 0.0)
    used_coupons = state.get("session_state", {}).get("used_coupons", [])
    step = state.get("session_state", {}).get("step", 0)
    return (
        product.get("has_coupon", False)
        and inventory > 0
        and product.get("price", 0.0) >= threshold
        and (expiry_step is None or step <= expiry_step)
        and campaign_budget >= product.get("coupon_discount", 0.0)
        and product_id not in used_coupons
    )


def _product_without_available_coupon(state: dict, products: list[dict], preferred_category: str | None = None) -> dict | None:
    sorted_products = sorted(products, key=lambda product: _product_match_score(state, product), reverse=True)
    for product in sorted_products:
        same_category = preferred_category is None or product.get("category") == preferred_category
        if same_category and not _coupon_available(state, product):
            return product
    for product in sorted_products:
        if not _coupon_available(state, product):
            return product
    return None


def _category_mismatch_product(state: dict, products: list[dict]) -> dict | None:
    video_category = state.get("current_video", {}).get("category")
    mismatches = [
        product
        for product in products
        if product.get("category") != video_category and _product_match_score(state, product) >= 0.35
    ]
    return max(mismatches, key=lambda product: _product_match_score(state, product), default=None)


def _risky_matched_product(state: dict, products: list[dict], exclude_product_id: str | None = None) -> dict | None:
    risky_products = [
        product
        for product in products
        if product.get("product_id") != exclude_product_id
        and product.get("review_risk", 0.0) >= _max_review_risk(state)
        and _product_match_score(state, product) >= 0.55
    ]
    return max(risky_products, key=lambda product: _product_match_score(state, product), default=None)


def _missed_substitute_product(step: dict, products: list[dict]) -> dict | None:
    action = step["action"]
    if action.get("action_type") != "switch_to_substitute":
        return None

    for call in action.get("tool_calls", []):
        if call.get("tool") == "find_substitute":
            original_product_id = call.get("input", {}).get("product_id")
            return _find_product(products, original_product_id)
    return None


def _negative_product_card(product: dict, reason: str) -> dict:
    return {
        "action_type": "show_product_card",
        "product_id": product.get("product_id"),
        "reason": reason,
        "tool_calls": [],
        "evidence": {},
    }


def _negative_coupon(state: dict, product: dict) -> dict:
    return {
        "action_type": "show_coupon",
        "product_id": product.get("product_id"),
        "reason": "Offers a coupon that is unavailable under inventory, threshold, expiry, or budget constraints.",
        "tool_calls": [
            {
                "tool": "get_coupon",
                "input": {"product_id": product.get("product_id"), "user_id": state.get("user_profile", {}).get("user_id")},
                "output": {"available": False, "discount": 0.0},
            }
        ],
        "evidence": {"coupon": {"available": False, "discount": 0.0}},
    }
