import json
import pytest
from videoshop.agents.adapters import ToolCallParseError, agent_step_from_json, agent_step_from_tool_calls
from videoshop.agents.llm_loop import FunctionCallingAgent, ScriptedToolCallingClient
from videoshop.agents.openai_compatible import OpenAICompatibleToolClient
from videoshop.agents.tool_specs import FINAL_ACTION_TOOL, video_shop_tool_specs
from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.simulator.env import VideoShopEnv


def _reasoning(decision_rule="Use grounded evidence."):
    return {
        "observation_facts": ["candidate is visible"],
        "evidence_used": [],
        "decision_rule": decision_rule,
        "confidence": 0.9,
    }


def test_tool_specs_include_business_tools_and_final_action():
    specs = video_shop_tool_specs()
    names = {spec["function"]["name"] for spec in specs}

    assert "get_coupon" in names
    assert "find_substitute" in names
    assert "explain_recommendation" in names
    assert FINAL_ACTION_TOOL in names
    final_spec = next(spec for spec in specs if spec["function"]["name"] == FINAL_ACTION_TOOL)
    assert "reasoning_summary" in final_spec["function"]["parameters"]["required"]


def test_openai_style_tool_calls_parse_to_agent_step():
    step = agent_step_from_tool_calls(
        [
            {"type": "function", "function": {"name": "get_coupon", "arguments": '{"product_id":"P001"}'}},
            {
                "type": "function",
                "function": {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": '{"action_type":"show_coupon","product_id":"P001","reason":"Valid coupon."}',
                },
            },
        ]
    )

    assert step.tool_requests[0].tool == "get_coupon"
    assert step.tool_requests[0].args == {"product_id": "P001"}
    assert step.final_action is not None
    assert step.final_action.action_type == "show_coupon"


def test_final_action_parses_reasoning_summary():
    step = agent_step_from_tool_calls(
        [
            {
                "name": FINAL_ACTION_TOOL,
                "arguments": {
                    "action_type": "show_product_card",
                    "product_id": "P001",
                    "reason": "Relevant product.",
                    "reasoning_summary": {
                        "observation_facts": ["current_video.category=home_organization"],
                        "evidence_used": ["visible candidate P001"],
                        "candidate_comparison": ["P001 matches the video category"],
                        "rejected_options": [],
                        "decision_rule": "Recommend relevant, in-stock product.",
                        "confidence": 0.8,
                    },
                },
            }
        ]
    )

    assert step.final_action is not None
    assert step.final_action.reasoning_summary["decision_rule"] == "Recommend relevant, in-stock product."


def test_final_action_missing_action_type_raises_parse_error():
    with pytest.raises(ToolCallParseError):
        agent_step_from_tool_calls(
            [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {"product_id": "P001", "reason": "Missing action type."},
                }
            ]
        )


def test_plain_json_agent_step_still_supported():
    step = agent_step_from_json(
        {
            "tool_requests": [{"tool": "explain_recommendation", "args": {"product_id": "P007"}}],
            "final_action": {"action_type": "show_explanation", "product_id": "P007", "reason": "High review risk."},
        }
    )

    assert step.tool_requests[0].tool == "explain_recommendation"
    assert step.final_action.action_type == "show_explanation"


def test_function_calling_agent_runs_tool_round_then_final_action():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)
    observation, info = env.reset_agent(seed=42)
    product_id = observation.visible_candidates[0]["product_id"]
    client = ScriptedToolCallingClient(
        [
            [{"name": "get_coupon", "arguments": {"product_id": product_id}}],
            [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {
                        "action_type": "show_coupon",
                        "product_id": product_id,
                        "reason": "Coupon was checked through the tool.",
                        "reasoning_summary": _reasoning("Show a verified coupon."),
                    },
                }
            ],
        ]
    )

    agent = FunctionCallingAgent(client, include_few_shots=False)
    step = agent.act(observation, info["state"])
    next_observation, reward, terminated, truncated, step_info = env.step_agent(step)

    assert step.tool_requests[0].tool == "get_coupon"
    assert step.final_action.action_type == "show_coupon"
    assert next_observation.session_summary["step"] == 1
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert step_info["violations"] == []
    assert [round_["status"] for round_ in step.metadata["tool_call_trace"]] == ["tools_executed", "accepted"]


def test_openai_compatible_client_posts_chat_completions(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {
                "id": "chatcmpl-test",
                "model": "Qwen3.8-27B",
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "get_coupon", "arguments": json.dumps({"product_id": "P001"})},
                                }
                            ]
                        }
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleToolClient(base_url="http://127.0.0.1:8001/v1", api_key="test-key", model="Qwen3.8-27B", timeout=3)

    calls = client.complete([{"role": "user", "content": "hi"}], video_shop_tool_specs())

    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert '"model": "Qwen3.8-27B"' in captured["payload"]
    assert calls[0]["function"]["name"] == "get_coupon"
    assert client.last_response_metadata["finish_reason"] == "tool_calls"
    assert client.last_response_metadata["usage"]["completion_tokens"] == 3
    assert client.last_response_metadata["latency_ms"] >= 0


def test_final_action_string_null_normalizes_to_none():
    step = agent_step_from_tool_calls(
        [
            {
                "name": FINAL_ACTION_TOOL,
                "arguments": {"action_type": "delay_recommendation", "product_id": "null", "reason": "Wait."},
            }
        ]
    )

    assert step.final_action.product_id is None


def test_mildly_malformed_tool_arguments_are_repaired():
    step = agent_step_from_tool_calls([{"name": "get_coupon", "arguments": "{'product_id': P001}"}])

    assert step.tool_requests[0].args == {"product_id": "P001"}


def test_function_calling_agent_falls_back_when_final_action_missing():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=1, seed=42)
    observation, info = env.reset_agent(seed=42)
    client = ScriptedToolCallingClient([[]])

    agent = FunctionCallingAgent(client, include_few_shots=False, max_tool_rounds=1)
    step = agent.act(observation, info["state"])
    _, _, _, truncated, step_info = env.step_agent(step)

    assert step.final_action.action_type == "delay_recommendation"
    assert "missing_final_action" in step.metadata["agent_warnings"]
    assert "missing_final_action" in step_info["violations"]
    assert truncated is True


def test_function_calling_agent_repairs_missing_final_action_once():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=1, seed=42)
    observation, info = env.reset_agent(seed=42)
    client = ScriptedToolCallingClient(
        [
            [],
            [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "Repaired final action.",
                        "reasoning_summary": _reasoning("Delay after an empty response."),
                    },
                }
            ],
        ]
    )

    agent = FunctionCallingAgent(client, include_few_shots=False, max_tool_rounds=2)
    step = agent.act(observation, info["state"])
    _, _, _, truncated, step_info = env.step_agent(step)

    assert step.final_action.action_type == "delay_recommendation"
    assert "agent_repair_warnings" in step.metadata
    assert "agent_warnings" not in step.metadata
    assert "missing_final_action" not in step_info["violations"]
    assert truncated is True


def test_function_calling_agent_falls_back_on_bad_tool_arguments():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=1, seed=42)
    observation, info = env.reset_agent(seed=42)
    client = ScriptedToolCallingClient([[{"name": "get_coupon", "arguments": "{bad json"}]])

    agent = FunctionCallingAgent(client, include_few_shots=False, max_tool_rounds=1)
    step = agent.act(observation, info["state"])
    _, _, _, truncated, step_info = env.step_agent(step)

    assert step.final_action.action_type == "delay_recommendation"
    assert "tool_call_parse_error" in step.metadata["agent_warnings"]
    assert "tool_call_parse_error" in step_info["violations"]
    assert truncated is True


def test_function_calling_agent_repairs_invalid_rank_arguments_before_env_step():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=1, seed=42)
    observation, info = env.reset_agent(seed=42)
    client = ScriptedToolCallingClient(
        [
            [{"name": "rank_products", "arguments": {"candidate_product_ids": '["P001"'}}],
            [
                {
                    "name": FINAL_ACTION_TOOL,
                    "arguments": {
                        "action_type": "delay_recommendation",
                        "product_id": None,
                        "reason": "Invalid ranking request was discarded.",
                        "reasoning_summary": _reasoning("Delay after discarding invalid tool arguments."),
                    },
                }
            ],
        ]
    )

    step = FunctionCallingAgent(client, include_few_shots=False, max_tool_rounds=2).act(
        observation, info["state"]
    )

    assert step.tool_requests == []
    assert step.final_action.action_type == "delay_recommendation"
    assert "tool_call_parse_error" in step.metadata["agent_repair_warnings"]
    assert "agent_warnings" not in step.metadata
