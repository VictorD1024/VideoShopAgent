"""LLM agent adapters for VideoShopEnv."""

from videoshop.agents.adapters import agent_step_from_json, agent_step_from_tool_calls
from videoshop.agents.llm_loop import FunctionCallingAgent
from videoshop.agents.openai_compatible import OpenAICompatibleToolClient
from videoshop.agents.prompt import build_system_prompt, observation_to_user_message
from videoshop.agents.tool_specs import video_shop_tool_specs

__all__ = [
    "FunctionCallingAgent",
    "OpenAICompatibleToolClient",
    "agent_step_from_json",
    "agent_step_from_tool_calls",
    "build_system_prompt",
    "observation_to_user_message",
    "video_shop_tool_specs",
]
