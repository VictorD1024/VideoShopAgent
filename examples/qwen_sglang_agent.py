from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.agents import FunctionCallingAgent, OpenAICompatibleToolClient
from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.simulator.env import VideoShopEnv


def main() -> None:
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)
    observation, info = env.reset_agent(seed=42)

    client = OpenAICompatibleToolClient(
        base_url=os.environ.get("VIDEOSHOP_LLM_BASE_URL"),
        api_key=os.environ.get("VIDEOSHOP_LLM_API_KEY"),
        model=os.environ.get("VIDEOSHOP_LLM_MODEL", "Qwen3.8-27B"),
    )
    agent = FunctionCallingAgent(client)
    step = agent.act(observation, info["state"])
    _, reward, terminated, truncated, step_info = env.step_agent(step)

    print(
        json.dumps(
            {
                "agent_step": asdict(step),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "violations": step_info["violations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
