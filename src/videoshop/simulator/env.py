from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

from videoshop.simulator.feed import VideoFeed
from videoshop.simulator.observation import build_observation
from videoshop.simulator.reward import compute_reward
from videoshop.simulator.scenario import VideoShopScenario
from videoshop.simulator.schemas import (
    AgentAction,
    AgentStep,
    EnvState,
    FinalAction,
    Observation,
    Product,
    StateUpdate,
    ToolCallResult,
    UserResponse,
    VideoContext,
)
from videoshop.simulator.state import apply_state_update
from videoshop.simulator.tool_runtime import ToolRuntime, tool_results_as_dicts, tool_results_to_calls
from videoshop.simulator.tools import product_match_score
from videoshop.simulator.user_simulator import UserSimulator

DEFAULT_TASK = "Assist a short-video shopping session with grounded, low-risk recommendations."
ACTION_TYPES = {
    "delay_recommendation",
    "show_product_card",
    "show_coupon",
    "switch_to_substitute",
    "show_explanation",
}
PRODUCT_ACTIONS = ACTION_TYPES - {"delay_recommendation"}


class VideoShopEnv:
    def __init__(
        self,
        initial_states: list[EnvState],
        videos: list[VideoContext],
        max_steps: int = 8,
        seed: int = 42,
        scenarios: list[VideoShopScenario] | None = None,
        observation_candidate_limit: int | None = None,
    ) -> None:
        if not initial_states:
            raise ValueError("VideoShopEnv requires at least one initial state.")
        self.initial_states = initial_states
        self.videos = videos
        self.scenarios = {scenario.scenario_id: scenario for scenario in scenarios or []}
        self.max_steps = max_steps
        self.seed = seed
        self.rng = random.Random(seed)
        self.feed = VideoFeed(videos, self.rng)
        self.user_simulator = UserSimulator(self.rng)
        self.tool_runtime = ToolRuntime()
        self.observation_candidate_limit = observation_candidate_limit
        self.state: EnvState | None = None
        self.current_task = DEFAULT_TASK
        self.last_step: dict[str, Any] | None = None

    def reset(self, scenario_id: str | None = None, seed: int | None = None) -> EnvState:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)

        if scenario_id is not None:
            scenario = self.scenarios.get(scenario_id)
            if scenario is None:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")
            self.state = scenario.to_initial_state()
            self.feed = VideoFeed(scenario.video_feed, self.rng)
            self.current_task = scenario.objective
        else:
            self.state = copy.deepcopy(self.rng.choice(self.initial_states))
            self.feed = VideoFeed(self.videos, self.rng)
            self.current_task = DEFAULT_TASK
        self.state.session_state.step = 0
        self.state.current_video = self.feed.sample_next(self.state.user_profile)
        self.last_step = None
        return self.state

    def step(self, action: AgentAction) -> tuple[EnvState, UserResponse, float, bool, StateUpdate]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        product = self.find_product(action.product_id)
        response = self.user_simulator.respond(self.state, action, product)
        category_match = product is None or product_match_score(self.state, product) >= 0.25
        reward = compute_reward(response, action, product, category_match=category_match)
        update = apply_state_update(self.state, action, response, product)
        done = update.episode_done or self.state.session_state.step >= self.max_steps

        if not done:
            self.state.current_video = self.feed.sample_next(self.state.user_profile)
            self.state.session_state.recent_watch_categories.append(self.state.current_video.category)

        return self.state, response, reward, done, update

    def reset_agent(
        self,
        *,
        seed: int | None = None,
        scenario_id: str | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        state = self.reset(scenario_id=scenario_id, seed=seed)
        info = {"state": state, "scenario_id": scenario_id, "seed": self.seed}
        return build_observation(state, task=self.current_task, candidate_limit=self.observation_candidate_limit), info

    def step_agent(
        self,
        step: AgentStep | AgentAction,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset_agent() before step_agent().")

        action, tool_results = self._normalize_agent_step(step)
        action_errors = self._action_shape_errors(action)
        if action_errors:
            raise ValueError("Invalid agent action: " + "; ".join(action_errors))
        violations = self.tool_runtime.validate_action_evidence(action, tool_results)
        if isinstance(step, AgentStep):
            violations.extend(step.metadata.get("agent_warnings", []))
        _, response, reward, done, update = self.step(action)
        del done
        terminated = update.episode_done
        truncated = self.state.session_state.step >= self.max_steps and not terminated

        self.last_step = {
            "tool_results": tool_results_as_dicts(tool_results),
            "action": asdict(action),
            "user_response": asdict(response),
            "reward": reward,
            "state_update": asdict(update),
            "violations": violations,
        }
        info = {
            "state": self.state,
            "user_response": response,
            "state_update": update,
            "tool_results": tool_results,
            "violations": violations,
        }
        observation = build_observation(
            self.state,
            task=self.current_task,
            last_step=self.last_step,
            candidate_limit=self.observation_candidate_limit,
        )
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "text") -> str:
        if self.state is None:
            return "VideoShopEnv(not reset)"
        if mode != "text":
            raise ValueError("Only text render mode is supported.")
        video = self.state.current_video
        session = self.state.session_state
        return (
            f"step={session.step} video={video.video_id} category={video.category} "
            f"skips={session.recent_skips} exposed={len(session.exposed_products)}"
        )

    def close(self) -> None:
        self.state = None
        self.last_step = None

    def find_product(self, product_id: str | None) -> Product | None:
        if self.state is None or product_id is None:
            return None
        for product in self.state.candidate_products:
            if product.product_id == product_id:
                return product
        return None

    def _normalize_agent_step(self, step: AgentStep | AgentAction) -> tuple[AgentAction, list[ToolCallResult]]:
        if isinstance(step, AgentAction):
            results = [
                ToolCallResult(tool=call.tool, input=call.input, output=call.output)
                for call in step.tool_calls
            ]
            return step, results

        tool_results = self.tool_runtime.execute_many(self.state, step.tool_requests)
        final = step.final_action or FinalAction(action_type="delay_recommendation", reason="No final action provided.")
        evidence = _evidence_from_tool_results(final, tool_results)
        action = AgentAction(
            action_type=final.action_type,
            product_id=final.product_id,
            reason=final.reason,
            tool_calls=tool_results_to_calls(tool_results),
            evidence=evidence,
        )
        return action, tool_results

    def _action_shape_errors(self, action: AgentAction) -> list[str]:
        errors: list[str] = []
        if action.action_type not in ACTION_TYPES:
            errors.append(f"unknown action_type: {action.action_type}")
        if action.action_type == "delay_recommendation" and action.product_id is not None:
            errors.append("delay_recommendation requires product_id=null")
        if action.action_type in PRODUCT_ACTIONS and not action.product_id:
            errors.append(f"{action.action_type} requires a product_id")
        if action.product_id and self.find_product(action.product_id) is None:
            errors.append(f"unknown product_id: {action.product_id}")
        return errors


def _evidence_from_tool_results(final: FinalAction, tool_results: list[ToolCallResult]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    successful_outputs = {result.tool: result.output for result in tool_results if result.success}

    if final.action_type == "show_coupon" and "get_coupon" in successful_outputs:
        evidence["coupon"] = successful_outputs["get_coupon"]
    if final.action_type == "switch_to_substitute" and "find_substitute" in successful_outputs:
        evidence["substitute"] = successful_outputs["find_substitute"]
    if final.action_type == "show_explanation" and "explain_recommendation" in successful_outputs:
        evidence.update(successful_outputs["explain_recommendation"].get("evidence", {}))

    return evidence
