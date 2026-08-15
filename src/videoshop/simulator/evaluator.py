from __future__ import annotations


def summarize_episodes(episodes: list[dict], policy_name: str) -> dict:
    if not episodes:
        return {"episodes": 0, "policy": policy_name}

    total_steps = sum(len(episode["steps"]) for episode in episodes)
    clicks = sum(step["user_response"]["clicked"] for episode in episodes for step in episode["steps"])
    carts = sum(step["user_response"]["added_to_cart"] for episode in episodes for step in episode["steps"])
    purchases = sum(episode["outcome"] == "purchase" for episode in episodes)
    interruptions = sum(step["user_response"]["interrupted"] for episode in episodes for step in episode["steps"])
    rewards = [episode["total_reward"] for episode in episodes]

    return {
        "episodes": len(episodes),
        "avg_reward": sum(rewards) / len(rewards),
        "avg_steps": total_steps / len(episodes),
        "ctr": clicks / total_steps if total_steps else 0.0,
        "add_to_cart_rate": carts / total_steps if total_steps else 0.0,
        "purchase_rate": purchases / len(episodes),
        "interruption_rate": interruptions / total_steps if total_steps else 0.0,
        "policy": policy_name,
    }
