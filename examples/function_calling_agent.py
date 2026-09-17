from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.agents.llm_loop import FunctionCallingAgent, ScriptedToolCallingClient
from videoshop.agents.tool_specs import FINAL_ACTION_TOOL
from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.simulator.env import VideoShopEnv


def main() -> None:
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)
    observation, info = env.reset_agent(seed=42)
    product_id = observation.visible_candidates[0]["product_id"]

    # Replace ScriptedToolCallingClient with a real OpenAI/Grok/Qwen client adapter.
    client = ScriptedToolCallingClient(
        [
            [{"name": "get_coupon", "arguments": {"product_id": product_id}}],
            [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {
                        "action_type": "show_coupon",
                        "product_id": product_id,
                        "reason": "Coupon availability was checked through the tool.",
                        "reasoning_summary": {
                            "observation_facts": [f"{product_id} is visible"],
                            "evidence_used": ["get_coupon verified coupon availability"],
                            "candidate_comparison": [],
                            "rejected_options": [],
                            "decision_rule": "Show a coupon only after successful verification.",
                            "confidence": 0.95,
                        },
                    },
                }
            ],
        ]
    )

    agent = FunctionCallingAgent(client, include_few_shots=False)
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
