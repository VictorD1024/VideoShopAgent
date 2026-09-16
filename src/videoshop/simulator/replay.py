from __future__ import annotations

from dataclasses import asdict

from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.schemas import AgentAction


def replay_episode(env: VideoShopEnv, episode: dict) -> dict:
    """Replay recorded final actions and compare recomputed rewards.

    The user simulator is stochastic, so exact user responses are not expected to match unless
    the caller recreates the original seed and scenario. This helper still validates action
    schema, termination behavior, and reward recomputation under the supplied env.
    """

    env.reset()
    mismatches: list[dict] = []
    replayed_steps = 0

    for recorded in episode.get("steps", []):
        action = AgentAction(**recorded["action"])
        _, response, reward, done, update = env.step(action)
        replayed_steps += 1

        if abs(float(recorded.get("reward", 0.0)) - reward) > 1e-9:
            mismatches.append(
                {
                    "t": recorded.get("t"),
                    "field": "reward",
                    "recorded": recorded.get("reward"),
                    "replayed": reward,
                    "replayed_response": asdict(response),
                    "replayed_update": asdict(update),
                }
            )

        if done:
            break

    return {
        "episode_id": episode.get("episode_id"),
        "replayed_steps": replayed_steps,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
