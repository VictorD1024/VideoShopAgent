from __future__ import annotations


def evaluate_episode(episode: dict) -> dict:
    steps = episode.get("steps", [])
    total_steps = len(steps)
    if total_steps == 0:
        return {
            "steps": 0,
            "total_reward": episode.get("total_reward", 0.0),
            "outcome": episode.get("outcome", "unknown"),
        }

    clicks = sum(bool(step["user_response"].get("clicked")) for step in steps)
    carts = sum(bool(step["user_response"].get("added_to_cart")) for step in steps)
    purchases = sum(bool(step["user_response"].get("purchased")) for step in steps)
    net_purchases = sum(
        bool(step["user_response"].get("purchased"))
        and not bool(step["user_response"].get("returned_or_refunded"))
        for step in steps
    )
    interruptions = sum(bool(step["user_response"].get("interrupted")) for step in steps)
    fake_coupon = sum(_fake_coupon(step) for step in steps)
    unsupported_explanation = sum(_unsupported_explanation(step) for step in steps)
    category_mismatch = sum(bool(step["user_response"].get("irrelevant_recommendation")) for step in steps)
    returns = sum(bool(step["user_response"].get("returned_or_refunded")) for step in steps)

    return {
        "steps": total_steps,
        "total_reward": episode.get("total_reward", 0.0),
        "outcome": episode.get("outcome", "unknown"),
        "ctr": clicks / total_steps,
        "add_to_cart_rate": carts / total_steps,
        "purchase_steps": purchases,
        "gross_purchase_steps": purchases,
        "net_purchase_steps": net_purchases,
        "gross_purchase": purchases > 0,
        "net_purchase": net_purchases > 0,
        "interruption_rate": interruptions / total_steps,
        "fake_coupon_count": fake_coupon,
        "unsupported_explanation_count": unsupported_explanation,
        "category_mismatch_count": category_mismatch,
        "return_or_refund_count": returns,
        "constraint_violations": fake_coupon + unsupported_explanation + category_mismatch,
    }


def _fake_coupon(step: dict) -> bool:
    action = step.get("action", {})
    evidence = action.get("evidence", {})
    coupon = evidence.get("coupon", {})
    return action.get("action_type") == "show_coupon" and not bool(coupon.get("available"))


def _unsupported_explanation(step: dict) -> bool:
    action = step.get("action", {})
    return action.get("action_type") == "show_explanation" and not bool(action.get("evidence"))
