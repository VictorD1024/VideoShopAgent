from __future__ import annotations

import random
from typing import Any

from videoshop.feed.control_env import FeedObservation
from videoshop.feed.schemas import FeedDecision
from videoshop.simulator.schemas import AgentAction, EnvState
from videoshop.simulator.tools import explain_recommendation, find_substitute, get_coupon


class RandomFeedPolicy:
    """Uniform choice over the whole candidate set, including ineligible ones.

    Keeping ineligible candidates in reach is intentional: the random floor should
    also measure how often the environment has to intercept an illegal exposure.
    """

    name = "random"

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, observation: FeedObservation) -> FeedDecision:
        candidate = self.rng.choice(observation.candidate_exposures)
        return FeedDecision.serve(candidate["exposure_id"], reason="Random baseline choice.")


class GreedyScorePolicy:
    """Serves whatever the underlying ranker predicts will earn the most GMV.

    This is the naive commerce-maximizing baseline. Because ``base_scores`` are
    commerce-optimistic, it over-exposes ads and should lose on user-weighted profiles.
    """

    name = "greedy_gmv"

    def act(self, observation: FeedObservation) -> FeedDecision:
        eligible = _eligible(observation.candidate_exposures) or observation.candidate_exposures
        best = max(eligible, key=lambda item: item["base_scores"]["expected_net_gmv"])
        return FeedDecision.serve(best["exposure_id"], reason="Highest predicted net GMV.")


class AlwaysOrganicPolicy:
    """Never intervenes commercially.

    The reference point for "do no commerce": any candidate policy has to beat this
    to justify showing a single product.
    """

    name = "always_organic"

    def act(self, observation: FeedObservation) -> FeedDecision:
        organic = [
            candidate
            for candidate in _eligible(observation.candidate_exposures)
            if candidate["source_type"] == "organic"
        ]
        pool = organic or _eligible(observation.candidate_exposures) or observation.candidate_exposures
        best = max(pool, key=lambda item: item["base_scores"]["expected_watch_time"])
        return FeedDecision.serve(best["exposure_id"], reason="Organic-only baseline.")


class RuleBasedFeedPolicy:
    """Density-controlled heuristic baseline.

    Respects hard constraints, caps commercial exposure density, backs off to organic
    content when the viewer is fatigued or skipping, and otherwise trades predicted
    commerce value against predicted skip risk.
    """

    name = "rule_based"

    def __init__(
        self,
        *,
        max_commercial_ratio: float = 0.4,
        fatigue_backoff: float = 0.6,
        skip_backoff: int = 3,
        skip_risk_weight: float = 2.0,
    ) -> None:
        self.max_commercial_ratio = max_commercial_ratio
        self.fatigue_backoff = fatigue_backoff
        self.skip_backoff = skip_backoff
        self.skip_risk_weight = skip_risk_weight

    def act(self, observation: FeedObservation) -> FeedDecision:
        eligible = _eligible(observation.candidate_exposures)
        if not eligible:
            # Nothing is servable; fall back so the episode stays inspectable.
            return FeedDecision.serve(
                observation.candidate_exposures[0]["exposure_id"],
                reason="No eligible candidate; falling back to the first candidate.",
            )

        organic = [item for item in eligible if item["source_type"] == "organic"]
        commercial = [item for item in eligible if item["source_type"] != "organic"]

        if self._should_back_off(observation) and organic:
            best = max(organic, key=self._content_score)
            return FeedDecision.serve(best["exposure_id"], reason="Backing off commerce to protect the session.")

        if not commercial or (organic and self._commercial_density(observation) >= self.max_commercial_ratio):
            pool = organic or eligible
            best = max(pool, key=self._content_score)
            return FeedDecision.serve(best["exposure_id"], reason="Commercial density cap reached.")

        best_commercial = max(commercial, key=self._commerce_score)
        best_organic = max(organic, key=self._content_score) if organic else None
        if best_organic is not None and self._content_score(best_organic) > self._commerce_score(best_commercial):
            return FeedDecision.serve(best_organic["exposure_id"], reason="Organic clip beats the commercial options.")
        return FeedDecision.serve(best_commercial["exposure_id"], reason="Commercial exposure is worth the interruption.")

    def _should_back_off(self, observation: FeedObservation) -> bool:
        user = observation.public_context.get("user_summary", {})
        session = observation.public_context.get("session_summary", {})
        return (
            float(user.get("ad_fatigue", 0.0)) >= self.fatigue_backoff
            or int(session.get("recent_skips", 0)) >= self.skip_backoff
        )

    def _commercial_density(self, observation: FeedObservation) -> float:
        """Share of the session so far that was commercial supply.

        Counts aired ad/seller/affiliate clips rather than exposed products: a
        commercial clip whose product treatment was declined carries no product but
        still spends the viewer's patience.
        """

        session = observation.public_context.get("session_summary", {})
        step = int(session.get("step", 0))
        if step <= 0:
            return 0.0
        return int(session.get("commercial_exposures", 0)) / step

    def _content_score(self, candidate: dict[str, Any]) -> float:
        scores = candidate["base_scores"]
        return scores["expected_watch_time"] / 20.0 - self.skip_risk_weight * scores["skip_probability"] * 0.5

    def _commerce_score(self, candidate: dict[str, Any]) -> float:
        scores = candidate["base_scores"]
        return (
            scores["expected_net_gmv"] / 10.0
            + scores["product_click_probability"]
            - self.skip_risk_weight * scores["skip_probability"]
            - scores["refund_probability"]
        )


class AnchorInterventionPolicy:
    """Intervention baseline for the composite environment.

    Unlike the v1 `RuleBasedPolicy`, which picks its own product, this one treats the
    exposure's anchor as given: the feed layer already sold the impression on that
    `video x product` pair. It only decides *how* to present it, and only leaves the
    anchor through `find_substitute`.
    """

    name = "anchor_rule_based"

    def __init__(self, *, risk_threshold: float = 0.35) -> None:
        self.risk_threshold = risk_threshold

    def act(self, observation, state: EnvState, exposure) -> AgentAction:
        del observation
        anchor = next(
            (item for item in state.candidate_products if item.product_id == exposure.product_id), None
        )
        if anchor is None:
            return AgentAction(action_type="delay_recommendation", reason="Anchor product is unavailable.")

        # No fatigue backoff here on purpose. The clip is already airing as ad or creator
        # supply and the intervention layer cannot change that, so dropping the product
        # pays the annoyance without the commerce. Backing off is the feed layer's call.
        tool_calls = []
        if anchor.inventory <= 0 or anchor.review_risk > 0.5:
            substitute, call = find_substitute(state, anchor)
            tool_calls.append(call)
            if substitute is None:
                return AgentAction(
                    action_type="delay_recommendation",
                    reason="Anchor is unsellable and no substitute exists.",
                    tool_calls=tool_calls,
                )
            return AgentAction(
                action_type="switch_to_substitute",
                product_id=substitute.product_id,
                reason="Anchor is out of stock or too risky.",
                tool_calls=tool_calls,
                evidence={"substitute": call.output},
            )

        if exposure.treatment == "coupon":
            call = get_coupon(state, anchor)
            tool_calls.append(call)
            if call.output["available"]:
                return AgentAction(
                    action_type="show_coupon",
                    product_id=anchor.product_id,
                    reason="The offered coupon checks out.",
                    tool_calls=tool_calls,
                    evidence={"coupon": call.output},
                )
            # The feed layer offered a coupon the catalog cannot back; degrade instead of faking it.

        if anchor.review_risk >= self.risk_threshold:
            call = explain_recommendation(state, anchor)
            tool_calls.append(call)
            return AgentAction(
                action_type="show_explanation",
                product_id=anchor.product_id,
                reason="Anchor carries review risk, so lead with evidence.",
                tool_calls=tool_calls,
                evidence=call.output["evidence"],
            )

        return AgentAction(
            action_type="show_product_card",
            product_id=anchor.product_id,
            reason="Anchor is in stock and low risk.",
            tool_calls=tool_calls,
        )


class TrustingInterventionPolicy:
    """Applies the offered treatment verbatim, without checking anything.

    The reference point for "no intervention layer": it shows whatever coupon the
    ranker advertised, so it walks into every stale-coupon trap.
    """

    name = "trusting"

    def act(self, observation, state: EnvState, exposure) -> AgentAction:
        del observation, state
        if exposure.treatment == "coupon":
            return AgentAction(
                action_type="show_coupon",
                product_id=exposure.product_id,
                reason="The ranker says there is a coupon.",
            )
        return AgentAction(
            action_type="show_product_card",
            product_id=exposure.product_id,
            reason="Show whatever was offered.",
        )


INTERVENTION_POLICIES = {
    AnchorInterventionPolicy.name: AnchorInterventionPolicy,
    TrustingInterventionPolicy.name: TrustingInterventionPolicy,
}


def build_intervention_policy(name: str):
    if name not in INTERVENTION_POLICIES:
        raise ValueError(
            f"Unknown intervention policy: {name}. Available: {sorted(INTERVENTION_POLICIES)}"
        )
    return INTERVENTION_POLICIES[name]()


def build_intervention_agent(client, **kwargs):
    """A `FunctionCallingAgent` briefed for the composite environment.

    The v1 system prompt tells the model to choose a product; here the feed layer has
    already chosen one, and the coupon is a claim to verify rather than a fact. Going
    through this factory is what keeps the two briefs from being mixed up.
    """

    from videoshop.agents.llm_loop import FunctionCallingAgent
    from videoshop.feed.observation import (
        build_intervention_system_prompt,
        intervention_few_shot_messages,
    )

    kwargs.setdefault("system_prompt", build_intervention_system_prompt())
    kwargs.setdefault("few_shots", intervention_few_shot_messages())
    return FunctionCallingAgent(client, **kwargs)


def _eligible(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if _is_eligible(candidate)]


def _is_eligible(candidate: dict[str, Any]) -> bool:
    eligibility = candidate["eligibility"]
    flags = ("in_stock", "coupon_valid", "policy_compliant", "risk_within_limit", "audience_allowed")
    return all(eligibility.get(flag, True) for flag in flags) and not eligibility.get("extra_blocks")


FEED_POLICIES = {
    "random": RandomFeedPolicy,
    "always_organic": AlwaysOrganicPolicy,
    "greedy_gmv": GreedyScorePolicy,
    "rule_based": RuleBasedFeedPolicy,
}


def build_feed_policy(name: str, seed: int = 0):
    if name == "random":
        return RandomFeedPolicy(random.Random(seed))
    try:
        return FEED_POLICIES[name]()
    except KeyError:
        raise ValueError(f"Unsupported feed policy: {name}. Available: {sorted(FEED_POLICIES)}") from None
