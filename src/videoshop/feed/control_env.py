from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from videoshop.feed.candidate_provider import CandidateBatch, ExposureTruth, SyntheticCandidateProvider
from videoshop.feed.reward import (
    DEFAULT_FEED_REWARD_CONFIG,
    FeedRewardConfig,
    RewardVector,
    compute_reward_vector,
    reward_terms,
)
from videoshop.feed.schemas import (
    COMMERCIAL_SOURCE_TYPES,
    FEED_SCHEMA_VERSION,
    ExposureCandidate,
    FeedDecision,
    ScenarioSpec,
)
from videoshop.feed.user_model import FeedUserModel, FeedUserResponse
from videoshop.simulator.schemas import Product, SessionState, UserProfile, VideoContext


@dataclass
class FeedObservation:
    """Everything the agent is allowed to see at one feed decision point."""

    schema_version: str
    scenario_id: str
    instruction: str
    step: int
    max_steps: int
    objective_profile: str
    public_context: dict[str, Any]
    candidate_exposures: list[dict[str, Any]]
    allowed_actions: list[str] = field(default_factory=lambda: ["serve_exposure"])
    last_step: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "instruction": self.instruction,
            "step": self.step,
            "max_steps": self.max_steps,
            "objective_profile": self.objective_profile,
            "public_context": copy.deepcopy(self.public_context),
            "candidate_exposures": copy.deepcopy(self.candidate_exposures),
            "allowed_actions": list(self.allowed_actions),
            "last_step": copy.deepcopy(self.last_step),
        }

    def exposure_ids(self) -> list[str]:
        return [candidate["exposure_id"] for candidate in self.candidate_exposures]


class FeedControlEnv:
    """Top-1 feed control loop.

    Each step the agent receives a bounded candidate set that always contains both
    organic and commercial exposures, and serves exactly one of them. Video/product
    bindings are candidate facts: the decision protocol only carries an exposure_id,
    so an agent structurally cannot re-bind a product to another video. Candidates
    failing a hard constraint are intercepted and never reach the user.
    """

    def __init__(
        self,
        provider: SyntheticCandidateProvider,
        *,
        max_steps: int = 6,
        seed: int = 42,
        user_model: FeedUserModel | None = None,
        reward_config: FeedRewardConfig | None = None,
    ) -> None:
        self.provider = provider
        self.max_steps = max_steps
        self.seed = seed
        self.user_model = user_model or FeedUserModel()
        self.reward_config = reward_config or DEFAULT_FEED_REWARD_CONFIG

        self.scenario: ScenarioSpec | None = None
        self.user: UserProfile | None = None
        self.session: SessionState | None = None
        self.rng = random.Random(seed)
        self.batch: CandidateBatch | None = None
        self.records: list[dict[str, Any]] = []
        self._last_step: dict[str, Any] | None = None

    def reset(self, scenario: ScenarioSpec, seed: int | None = None) -> FeedObservation:
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self._assert_same_world(scenario)
        self.scenario = scenario
        self.user = _user_from_hidden_state(scenario)
        self.session = _session_from_hidden_state(scenario)
        self.records = []
        self._last_step = None
        self.batch = _batch_from_scenario(scenario)
        return self._observation()

    def step(self, decision: FeedDecision) -> tuple[FeedObservation, RewardVector, bool, bool, dict[str, Any]]:
        if self.scenario is None or self.batch is None or self.user is None or self.session is None:
            raise RuntimeError("Call reset() before step().")

        candidate = self.batch_candidate(decision.exposure_id)
        block_reasons = self._block_reasons(candidate)

        if block_reasons:
            vector = RewardVector(risk_cost=self.reward_config.invalid_decision_cost)
            record = self._record(decision, candidate, None, vector, {}, block_reasons)
            self.session.step += 1
            terminated = False
        else:
            assert candidate is not None
            offered = candidate
            candidate, truth, intervention = self._realize_exposure(candidate)
            violations = list(intervention.get("violations", ()))
            response = self.user_model.respond(candidate, truth, self.user, self.session, self.rng)
            terms = reward_terms(candidate, truth, response, self.user, self.reward_config, violations)
            vector = compute_reward_vector(
                candidate, truth, response, self.user, self.reward_config, violations
            )
            video = self.batch.videos.get(candidate.video_id)
            update = self.user_model.apply(
                response,
                candidate,
                truth,
                self.user,
                self.session,
                video.category if video else "unknown",
            )
            record = self._record(decision, candidate, response, vector, terms, [])
            record["offered_exposure"] = offered.to_dict()
            record["state_update"] = update.to_dict()
            if intervention:
                record["intervention"] = intervention
            terminated = update.session_ended

        self.records.append(record)
        self._last_step = _public_last_step(record)
        truncated = not terminated and self.session.step >= self.max_steps

        if not terminated and not truncated:
            self.batch = self.provider.generate(
                self.user,
                self.session,
                self.session.step,
                self.rng,
                exposure_prefix=_exposure_prefix(self.scenario.scenario_id),
            )

        info = {
            "record": record,
            "blocked": bool(block_reasons),
            "block_reasons": block_reasons,
            "reward_terms": record.get("reward_terms", {}),
            "scalar_reward": record["scalar_reward"],
        }
        return self._observation(), vector, terminated, truncated, info

    def _assert_same_world(self, scenario: ScenarioSpec) -> None:
        """Refuse a scenario that was not built from this provider's entities.

        Step 0 comes from the scenario and later steps come from the provider. If they
        disagree, an episode silently mixes two catalogs that share ids but not
        attributes, which is exactly what split-level entity isolation is meant to stop.
        """

        self._assert_entities_match(
            scenario,
            "product",
            scenario.hidden_world_state.get("products", {}),
            {product.product_id: product for product in self.provider.products},
        )
        self._assert_entities_match(
            scenario,
            "video",
            scenario.hidden_world_state.get("videos", {}),
            {video.video_id: video for video in self.provider.videos},
        )

    @staticmethod
    def _assert_entities_match(
        scenario: ScenarioSpec,
        kind: str,
        stored: dict[str, Any],
        catalog: dict[str, Any],
    ) -> None:
        missing = sorted(set(stored) - set(catalog))
        if missing:
            raise ValueError(
                f"Scenario {scenario.scenario_id} references {kind}s the provider does not own: "
                f"{missing[:5]}. Build scenarios and provider from the same split."
            )
        for entity_id, payload in stored.items():
            frozen = _entity_fingerprint(payload)
            live = _entity_fingerprint(asdict(catalog[entity_id]))
            if frozen == live:
                continue
            differing = sorted(
                f"{field} ({frozen.get(field)!r} vs {live.get(field)!r})"
                for field in set(frozen) | set(live)
                if frozen.get(field) != live.get(field)
            )
            raise ValueError(
                f"Scenario {scenario.scenario_id} and the provider disagree about {kind} "
                f"{entity_id} on {differing}. Two different catalogs are in play."
            )

    def _realize_exposure(
        self, candidate: ExposureCandidate
    ) -> tuple[ExposureCandidate, ExposureTruth, dict[str, Any]]:
        """Decide what is actually shown for a chosen candidate.

        The feed-only environment serves the offered exposure verbatim: the treatment
        is part of the candidate. :class:`~videoshop.feed.composite_env.FeedInterventionEnv`
        overrides this to let the intervention layer choose the treatment with tools.
        """

        assert self.batch is not None
        return candidate, self.batch.truth[candidate.exposure_id], {}

    def batch_candidate(self, exposure_id: str) -> ExposureCandidate | None:
        if self.batch is None:
            return None
        for candidate in self.batch.candidates:
            if candidate.exposure_id == exposure_id:
                return candidate
        return None

    def trajectory(self, *, episode_id: str, policy: str) -> dict[str, Any]:
        if self.scenario is None:
            raise RuntimeError("Call reset() before trajectory().")
        total = RewardVector.zero()
        for record in self.records:
            total = total + RewardVector.from_dict(record["reward_vector"])
        return {
            "schema_version": FEED_SCHEMA_VERSION,
            "episode_id": episode_id,
            "scenario_id": self.scenario.scenario_id,
            "policy": policy,
            "seed": self.seed,
            "objective_profile": self.scenario.reward_config.profile,
            "steps": copy.deepcopy(self.records),
            "total_reward_vector": total.to_dict(),
            "total_scalar_reward": round(self.scenario.reward_config.scalarize(total), 6),
        }

    def _block_reasons(self, candidate: ExposureCandidate | None) -> list[str]:
        if candidate is None:
            return ["unknown_exposure_id"]
        return list(candidate.eligibility.blocking_reasons())

    def _record(
        self,
        decision: FeedDecision,
        candidate: ExposureCandidate | None,
        response: FeedUserResponse | None,
        vector: RewardVector,
        terms: dict[str, dict[str, float]],
        block_reasons: list[str],
    ) -> dict[str, Any]:
        assert self.scenario is not None and self.session is not None and self.user is not None
        return {
            "schema_version": FEED_SCHEMA_VERSION,
            "t": self.session.step,
            "candidate_exposure_ids": [item.exposure_id for item in (self.batch.candidates if self.batch else [])],
            "decision": decision.to_dict(),
            "served_exposure": candidate.to_dict() if candidate and not block_reasons else None,
            "blocked": bool(block_reasons),
            "block_reasons": block_reasons,
            "user_response": response.to_dict() if response else None,
            "reward_vector": vector.to_dict(),
            "reward_terms": terms,
            "scalar_reward": round(self.scenario.reward_config.scalarize(vector), 6),
            "user_state_after": {
                "ad_fatigue": round(self.user.ad_fatigue, 4),
                "recent_skips": self.session.recent_skips,
            },
        }

    def _commercial_exposures(self) -> int:
        """How much commercial supply the session has actually aired.

        Not the same as `exposed_products`: an ad whose product treatment was declined
        still interrupted the viewer, and density control has to see it.
        """

        return sum(
            1
            for record in self.records
            if (record.get("served_exposure") or {}).get("source_type") in COMMERCIAL_SOURCE_TYPES
        )

    def _observation(self) -> FeedObservation:
        assert self.scenario is not None and self.batch is not None
        assert self.user is not None and self.session is not None

        public_context = copy.deepcopy(self.scenario.public_context)
        public_context["user_summary"] = _user_summary(self.user)
        public_context["session_summary"] = _session_summary(
            self.session, commercial_exposures=self._commercial_exposures()
        )
        public_context["catalog"] = self.batch.public_catalog()

        return FeedObservation(
            schema_version=FEED_SCHEMA_VERSION,
            scenario_id=self.scenario.scenario_id,
            instruction=self.scenario.instruction,
            step=self.session.step,
            max_steps=self.max_steps,
            objective_profile=self.scenario.reward_config.profile,
            public_context=public_context,
            candidate_exposures=[candidate.to_dict() for candidate in self.batch.candidates],
            last_step=copy.deepcopy(self._last_step),
        )


def _public_last_step(record: dict[str, Any]) -> dict[str, Any]:
    """Feedback the agent may see. Reward terms stay in the trajectory, not the observation."""
    response = record.get("user_response") or {}
    return {
        "t": record["t"],
        "served_exposure_id": (record.get("served_exposure") or {}).get("exposure_id"),
        "blocked": record["blocked"],
        "block_reasons": list(record["block_reasons"]),
        "observed_response": {
            key: response[key]
            for key in ("watched_seconds", "skipped", "clicked", "added_to_cart", "purchased", "refunded")
            if key in response
        },
    }


def _exposure_prefix(scenario_id: str) -> str:
    return f"{scenario_id}:"


def _batch_from_scenario(scenario: ScenarioSpec) -> CandidateBatch:
    raw_truth = scenario.hidden_world_state.get("exposure_truth", {})
    raw_videos = scenario.hidden_world_state.get("videos", {})
    raw_products = scenario.hidden_world_state.get("products", {})
    missing = [
        candidate.exposure_id
        for candidate in scenario.candidate_exposures
        if candidate.exposure_id not in raw_truth
    ]
    if missing:
        raise ValueError(
            f"Scenario {scenario.scenario_id} is missing hidden exposure truth for: {missing}"
        )
    return CandidateBatch(
        candidates=list(scenario.candidate_exposures),
        truth={key: ExposureTruth(**value) for key, value in raw_truth.items()},
        videos={key: VideoContext(**value) for key, value in raw_videos.items()},
        products={key: Product(**value) for key, value in raw_products.items()},
    )


def _user_from_hidden_state(scenario: ScenarioSpec) -> UserProfile:
    payload = scenario.hidden_world_state.get("user_profile")
    if payload is None:
        raise ValueError(f"Scenario {scenario.scenario_id} requires hidden_world_state.user_profile.")
    return UserProfile(**copy.deepcopy(payload))


def _session_from_hidden_state(scenario: ScenarioSpec) -> SessionState:
    payload = scenario.hidden_world_state.get("session_state") or {}
    return SessionState(**copy.deepcopy(payload))


def _user_summary(user: UserProfile) -> dict[str, Any]:
    """Redacted user view. purchase_intent stays hidden: it is a response parameter."""
    return {
        "user_id": user.user_id,
        "country": user.country,
        "budget_level": user.budget_level,
        "style_preferences": list(user.style_preferences),
        "category_interests": dict(user.category_interests),
        "price_sensitivity": user.price_sensitivity,
        "risk_sensitivity": user.risk_sensitivity,
        "ad_fatigue": round(user.ad_fatigue, 4),
    }


def _entity_fingerprint(payload: Any) -> dict[str, Any]:
    """Normalise an entity payload so a frozen scenario can be compared field by field.

    Scenarios are stored as plain dicts and may have made a JSON round trip, which turns
    tuples into lists and can nudge floats, so only those two need smoothing.
    """

    return {key: _normalise(value) for key, value in dict(payload).items()}


def _normalise(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in sorted(value.items())}
    return value


def _session_summary(session: SessionState, *, commercial_exposures: int = 0) -> dict[str, Any]:
    return {
        "step": session.step,
        "commercial_exposures": commercial_exposures,
        "recent_watch_categories": list(session.recent_watch_categories),
        "recent_clicks": list(session.recent_clicks),
        "recent_carts": list(session.recent_carts),
        "recent_skips": session.recent_skips,
        "exposed_products": list(session.exposed_products),
        "used_coupons": list(session.used_coupons),
    }
