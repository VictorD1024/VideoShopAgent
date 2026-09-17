from __future__ import annotations

from dataclasses import asdict

from videoshop.simulator.env import VideoShopEnv


def run_episode(env: VideoShopEnv, policy, episode_id: str) -> dict:
    state = env.reset()
    steps = []
    total_reward = 0.0
    done = False
    outcome = "no_purchase"

    while not done:
        state_snapshot = asdict(state)
        action = policy.act(state)
        next_state, response, reward, done, update = env.step(action)
        total_reward += reward
        if response.purchased:
            outcome = "returned_or_refunded" if response.returned_or_refunded else "purchase"

        steps.append(
            {
                "t": next_state.session_state.step,
                "state": state_snapshot,
                "tool_calls": [asdict(call) for call in action.tool_calls],
                "action": asdict(action),
                "user_response": asdict(response),
                "reward": reward,
                "state_update": asdict(update),
            }
        )
        state = next_state

    return {
        "episode_id": episode_id,
        "user_profile": asdict(state.user_profile),
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }


def run_batch(env: VideoShopEnv, policy, episodes: int) -> list[dict]:
    return [run_episode(env, policy, f"E{idx + 1:06d}") for idx in range(episodes)]
