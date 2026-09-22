from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import asdict
from typing import Any, Protocol

from videoshop.feed.candidate_provider import ExposureTruth
from videoshop.feed.control_env import FeedControlEnv
from videoshop.feed.legacy import treatment_for_legacy_action
from videoshop.feed.observation import (
    INTERVENTION_TASK,
    InterventionObservation,
    build_intervention_observation,
)
from videoshop.feed.schemas import ExposureCandidate
from videoshop.simulator.schemas import (
    AgentAction,
    AgentStep,
    CommerceContext,
    EnvState,
    FinalAction,
    Product,
    ToolCallResult,
)
from videoshop.simulator.tool_runtime import ToolRuntime, tool_results_as_dicts
from videoshop.simulator.tools import coupon_is_available

INTERVENTION_POOL_SIZE = 6


class InterventionPolicy(Protocol):
    """Either a v1 state policy (``act(state)``) or a tool-calling agent (``act(observation, state)``)."""

    def act(self, *args: Any) -> AgentAction | AgentStep: ...


class FeedInterventionEnv(FeedControlEnv):
    """The two-layer environment: feed chooses the exposure, intervention chooses the treatment.

    The feed policy picks a `user x video x product` exposure. For commercial
    exposures the intervention layer then runs the real v1 machinery, `ToolRuntime`
    plus evidence validation, to decide *how* to present it. The candidate's
    ``treatment`` becomes an offer rather than a fact: the intervention agent can
    downgrade a coupon offer it cannot substantiate, and the realized treatment is
    what the user actually responds to.
    """

    def __init__(
        self,
        provider,
        intervention_policy: InterventionPolicy,
        *,
        max_steps: int = 6,
        seed: int = 42,
        user_model=None,
        reward_config=None,
        intervention_pool_size: int = INTERVENTION_POOL_SIZE,
        intervention_task: str = INTERVENTION_TASK,
    ) -> None:
        super().__init__(
            provider,
            max_steps=max_steps,
            seed=seed,
            user_model=user_model,
            reward_config=reward_config,
        )
        self.intervention_policy = _brief_intervention_policy(intervention_policy)
        self.intervention_pool_size = intervention_pool_size
        self.intervention_task = intervention_task
        self.tool_runtime = ToolRuntime()

    def _realize_exposure(
        self, candidate: ExposureCandidate
    ) -> tuple[ExposureCandidate, ExposureTruth, dict[str, Any]]:
        assert self.batch is not None and self.user is not None and self.session is not None

        truth = self.batch.truth[candidate.exposure_id]
        if candidate.is_organic:
            return candidate, truth, {"ran": False, "reason": "organic_exposure_needs_no_treatment"}

        state = self._intervention_state(candidate, truth)
        if state is None:
            return candidate, truth, {"ran": False, "reason": "anchor_product_unavailable"}

        action, tool_results, warnings, agent_step = self._run_intervention(state, candidate)
        violations = self.tool_runtime.validate_action_evidence(action, tool_results)
        violations.extend(warnings)

        realized, extra_violations = self._apply_action(candidate, action, state, tool_results)
        violations.extend(extra_violations)

        product = _product_by_id(state, realized.product_id) if realized.product_id else None
        video = self.batch.videos.get(realized.video_id)
        realized_truth = (
            self.provider.truth_for(
                self.user, self.session, video, product, realized.source_type, realized.treatment
            )
            if video is not None
            else truth
        )

        return (
            realized,
            realized_truth,
            {
                "ran": True,
                "action": asdict(action),
                # The unabridged agent turn: tool_requests, reasoning_summary,
                # tool_call_trace, provider metadata and repair warnings. Without it the
                # episode can be scored but not used as a training trajectory.
                "agent_step": agent_step,
                "offered_treatment": candidate.treatment,
                "realized_treatment": realized.treatment,
                "tool_results": tool_results_as_dicts(tool_results),
                "violations": violations,
                "intervention_candidate_ids": [item.product_id for item in state.candidate_products],
                "observation": asdict(self._intervention_observation(state, candidate)),
            },
        )

    def _intervention_observation(
        self, state: EnvState, exposure: ExposureCandidate
    ) -> InterventionObservation:
        return build_intervention_observation(
            state,
            exposure,
            task=self.intervention_task,
            last_step=copy.deepcopy(self._last_step),
        )

    def _run_intervention(
        self, state: EnvState, exposure: ExposureCandidate
    ) -> tuple[AgentAction, list[ToolCallResult], list[str], dict[str, Any] | None]:
        observation = self._intervention_observation(state, exposure)
        result = self._call_policy(observation, state, exposure)

        if isinstance(result, AgentAction):
            results = [
                ToolCallResult(tool=call.tool, input=call.input, output=call.output)
                for call in result.tool_calls
            ]
            return result, results, [], None

        tool_results = self.tool_runtime.execute_many(state, result.tool_requests)
        final = result.final_action or FinalAction(
            action_type="delay_recommendation", reason="No final action provided."
        )
        action = AgentAction(
            action_type=final.action_type,
            product_id=final.product_id,
            reason=final.reason,
            tool_calls=[],
            evidence=_evidence_from_tool_results(final, tool_results),
        )
        return action, tool_results, list(result.metadata.get("agent_warnings", [])), _agent_step_record(result, self.intervention_policy)

    def _call_policy(
        self, observation, state: EnvState, exposure: ExposureCandidate
    ) -> AgentAction | AgentStep:
        """Dispatch on policy arity.

        ``act(state)``                       v1 policies
        ``act(observation, state)``          tool-calling agents (FunctionCallingAgent)
        ``act(observation, state, exposure)``anchor-aware intervention policies

        The arity is inspected rather than probed with try/except so that a TypeError
        raised inside the policy is not mistaken for a signature mismatch.
        """

        act = self.intervention_policy.act
        try:
            arity = len(inspect.signature(act).parameters)
        except (TypeError, ValueError):
            arity = 1
        if arity >= 3:
            return act(observation, state, exposure)
        if arity == 2:
            return act(observation, state)
        return act(state)

    def _apply_action(
        self,
        candidate: ExposureCandidate,
        action: AgentAction,
        state: EnvState,
        tool_results: list[ToolCallResult],
    ) -> tuple[ExposureCandidate, list[str]]:
        """Turn a v1 final action into the exposure that is actually served."""

        violations: list[str] = []
        treatment = treatment_for_legacy_action(action.action_type)
        product_id = action.product_id

        if treatment == "none" or not product_id:
            # The intervention layer declined the product. The clip still airs and is
            # still ad or creator supply; declining must not launder it into organic.
            return (
                ExposureCandidate(
                    exposure_id=candidate.exposure_id,
                    video_id=candidate.video_id,
                    source_type=candidate.source_type,
                    treatment="none",
                    placement=candidate.placement,
                    product_id=None,
                    base_scores=candidate.base_scores,
                    eligibility=candidate.eligibility,
                ),
                violations,
            )

        product = _product_by_id(state, product_id)
        if product is None:
            violations.append("product_outside_intervention_candidates")
            product_id = candidate.product_id
            product = _product_by_id(state, product_id) if product_id else None

        video = self.batch.videos.get(candidate.video_id) if self.batch else None
        if product is not None and video is not None and product.category != video.category:
            # The feed layer fixed the video; the intervention layer may not drift off it.
            violations.append("intervention_rebound_product_to_foreign_category")
            product_id = candidate.product_id
            product = _product_by_id(state, product_id) if product_id else None

        if treatment == "coupon" and product is not None and not coupon_is_available(state, product):
            violations.append("fake_coupon")
            treatment = "product_anchor"

        if product_id != candidate.product_id and not _substitution_is_grounded(
            action, candidate, product_id, tool_results
        ):
            # Swapping the anchor is only legitimate through the substitute tool, and
            # only to the product that tool actually returned for this anchor.
            violations.append("intervention_substituted_without_substitute_tool")

        return (
            ExposureCandidate(
                exposure_id=candidate.exposure_id,
                video_id=candidate.video_id,
                source_type=candidate.source_type,
                treatment=treatment,
                placement=candidate.placement,
                product_id=product_id,
                base_scores=candidate.base_scores,
                eligibility=candidate.eligibility,
            ),
            violations,
        )

    def _intervention_state(self, candidate: ExposureCandidate, truth: ExposureTruth) -> EnvState | None:
        """Build the v1 EnvState this exposure hands down to the intervention layer."""

        assert self.batch is not None and self.user is not None and self.session is not None
        video = self.batch.videos.get(candidate.video_id)
        anchor = self._product(candidate.product_id)
        if video is None or anchor is None:
            return None

        pool = [anchor]
        for product in sorted(self.provider.products, key=lambda item: item.product_id):
            if len(pool) >= self.intervention_pool_size:
                break
            if product.category == video.category and product.product_id != anchor.product_id:
                pool.append(product)

        pool = copy.deepcopy(pool)
        return EnvState(
            user_profile=self.user,
            session_state=self.session,
            current_video=video,
            candidate_products=pool,
            commerce_context=_commerce_context(pool, anchor, candidate, truth),
        )

    def _product(self, product_id: str | None) -> Product | None:
        if product_id is None:
            return None
        if self.batch is not None and product_id in self.batch.products:
            return self.batch.products[product_id]
        for product in self.provider.products:
            if product.product_id == product_id:
                return product
        return None


def _brief_intervention_policy(policy: InterventionPolicy) -> InterventionPolicy:
    """A bare `FunctionCallingAgent` still carries the v1 brief.

    Dropping one into the composite env without going through `build_intervention_agent`
    would teach the model to pick a product and to look for coupon inventory in the
    observation. Rewrite the default v1 prompt and few-shots; leave an already-custom
    brief alone.
    """

    from videoshop.agents.llm_loop import FunctionCallingAgent
    from videoshop.agents.prompt import build_system_prompt
    from videoshop.feed.observation import (
        build_intervention_system_prompt,
        intervention_few_shot_messages,
    )

    if not isinstance(policy, FunctionCallingAgent):
        return policy
    if policy.system_prompt == build_system_prompt():
        policy.system_prompt = build_intervention_system_prompt()
        policy.few_shots = intervention_few_shot_messages()
    return policy


def _agent_step_record(step: AgentStep, policy: InterventionPolicy) -> dict[str, Any]:
    """Keep the turn in a form that can be trained on, not just scored."""

    payload = asdict(step)
    metadata = payload.setdefault("metadata", {})
    system_prompt = getattr(policy, "system_prompt", None)
    if system_prompt:
        metadata["system_prompt"] = system_prompt
    return payload


def _commerce_context(
    products: list[Product],
    anchor: Product,
    candidate: ExposureCandidate,
    truth: ExposureTruth,
) -> CommerceContext:
    """Deterministic coupon state, authoritative where the candidate metadata is not.

    A coupon exposure can pass feed-level eligibility and still be a trap: the truth
    decides whether it redeems, and only `get_coupon` reveals that. This is what gives
    the intervention layer something to do that the feed layer structurally cannot.
    """

    inventory: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    expiry: dict[str, int] = {}

    for product in products:
        if not product.has_coupon:
            continue
        thresholds[product.product_id] = round(product.price * 0.5, 2)
        expiry[product.product_id] = 999
        if product.product_id == anchor.product_id and candidate.treatment == "coupon":
            inventory[product.product_id] = 50 if truth.coupon_available else 0
        else:
            # Stable pseudo-inventory: same product, same answer, every step and every
            # process. Python's hash() is salted per interpreter by PYTHONHASHSEED.
            inventory[product.product_id] = 40 if _stable_hash(product.product_id) % 3 else 0

    return CommerceContext(
        coupon_inventory=inventory,
        coupon_thresholds=thresholds,
        coupon_expiry_steps=expiry,
        campaign_budget=1000.0,
        stock_pressure={product.product_id: 0.5 for product in products if product.is_clearance},
        risk_constraints={"max_review_risk": 0.55},
    )


def _stable_hash(value: str) -> int:
    """Process-independent hash, so a frozen eval replays the same way anywhere."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _substitution_is_grounded(
    action: AgentAction,
    candidate: ExposureCandidate,
    product_id: str | None,
    tool_results: list[ToolCallResult],
) -> bool:
    """A swap counts only if a `find_substitute` on this anchor returned this product."""

    if action.action_type != "switch_to_substitute":
        return False
    return any(
        result.success
        and result.tool == "find_substitute"
        and result.input.get("product_id") == candidate.product_id
        and result.output.get("product_id") == product_id
        for result in tool_results
    )


def _product_by_id(state: EnvState, product_id: str | None) -> Product | None:
    if product_id is None:
        return None
    for product in state.candidate_products:
        if product.product_id == product_id:
            return product
    return None


def _evidence_from_tool_results(final: FinalAction, tool_results: list[ToolCallResult]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    outputs = {result.tool: result.output for result in tool_results if result.success}

    if final.action_type == "show_coupon" and "get_coupon" in outputs:
        evidence["coupon"] = outputs["get_coupon"]
    if final.action_type == "switch_to_substitute" and "find_substitute" in outputs:
        evidence["substitute"] = outputs["find_substitute"]
    if final.action_type == "show_explanation" and "explain_recommendation" in outputs:
        evidence.update(outputs["explain_recommendation"].get("evidence", {}))
    return evidence
