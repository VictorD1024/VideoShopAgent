from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from videoshop.feed.reward import RewardConfig

FEED_SCHEMA_VERSION = "v2"
"""Schema version for feed-control trajectories. v1 is the frozen legacy intervention schema."""

LEGACY_SCHEMA_VERSION = "v1"

SOURCE_TYPES = ("organic", "seller", "affiliate", "ad")
TREATMENTS = ("none", "product_anchor", "coupon")
PLACEMENTS = ("for_you", "search", "shop_tab")

COMMERCIAL_SOURCE_TYPES = ("seller", "affiliate", "ad")

PROBABILITY_SCORES = (
    "skip_probability",
    "product_click_probability",
    "purchase_probability",
    "refund_probability",
)


@dataclass(frozen=True)
class BaseScores:
    """Imperfect predictions from the underlying recommender.

    These are deliberately noisy: they are the ranker's beliefs, not ground truth.
    The true response parameters live in ``ScenarioSpec.hidden_world_state``.
    """

    expected_watch_time: float = 0.0
    skip_probability: float = 0.0
    product_click_probability: float = 0.0
    purchase_probability: float = 0.0
    expected_net_gmv: float = 0.0
    refund_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.expected_watch_time < 0.0:
            raise ValueError("BaseScores.expected_watch_time must be non-negative.")
        if self.expected_net_gmv < 0.0:
            raise ValueError("BaseScores.expected_net_gmv must be non-negative.")
        for name in PROBABILITY_SCORES:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"BaseScores.{name} must be within [0, 1], got {value}.")

    def to_dict(self) -> dict[str, float]:
        return {
            "expected_watch_time": self.expected_watch_time,
            "skip_probability": self.skip_probability,
            "product_click_probability": self.product_click_probability,
            "purchase_probability": self.purchase_probability,
            "expected_net_gmv": self.expected_net_gmv,
            "refund_probability": self.refund_probability,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaseScores":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown BaseScores fields: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in payload.items()})


@dataclass(frozen=True)
class Eligibility:
    """Hard constraints the environment enforces before an exposure can be served."""

    in_stock: bool = True
    coupon_valid: bool = True
    policy_compliant: bool = True
    risk_within_limit: bool = True
    audience_allowed: bool = True
    extra_blocks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_blocks", tuple(self.extra_blocks))

    @property
    def is_eligible(self) -> bool:
        return not self.blocking_reasons()

    def blocking_reasons(self) -> tuple[str, ...]:
        reasons = [
            name
            for name in ("in_stock", "coupon_valid", "policy_compliant", "risk_within_limit", "audience_allowed")
            if not getattr(self, name)
        ]
        reasons.extend(self.extra_blocks)
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_stock": self.in_stock,
            "coupon_valid": self.coupon_valid,
            "policy_compliant": self.policy_compliant,
            "risk_within_limit": self.risk_within_limit,
            "audience_allowed": self.audience_allowed,
            "extra_blocks": list(self.extra_blocks),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Eligibility":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown Eligibility fields: {sorted(unknown)}")
        data = dict(payload)
        extra = data.pop("extra_blocks", ())
        return cls(**{key: bool(value) for key, value in data.items()}, extra_blocks=tuple(extra))


@dataclass(frozen=True)
class ExposureCandidate:
    """One servable ``user x video x product x treatment`` unit.

    The video/product binding is a candidate fact produced by the environment.
    Agents select among candidates; they never re-bind a product to another video.
    """

    exposure_id: str
    video_id: str
    source_type: str
    treatment: str = "none"
    placement: str = "for_you"
    product_id: str | None = None
    base_scores: BaseScores = field(default_factory=BaseScores)
    eligibility: Eligibility = field(default_factory=Eligibility)

    def __post_init__(self) -> None:
        if not self.exposure_id:
            raise ValueError("ExposureCandidate.exposure_id is required.")
        if not self.video_id:
            raise ValueError("ExposureCandidate.video_id is required.")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"Unknown source_type: {self.source_type}. Allowed: {list(SOURCE_TYPES)}")
        if self.treatment not in TREATMENTS:
            raise ValueError(f"Unknown treatment: {self.treatment}. Allowed: {list(TREATMENTS)}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"Unknown placement: {self.placement}. Allowed: {list(PLACEMENTS)}")
        if self.source_type == "organic":
            if self.treatment != "none":
                raise ValueError("organic exposures must use treatment=none.")
            if self.product_id:
                raise ValueError("organic exposures must not carry a product_id.")
        elif self.treatment == "none":
            # A commercial clip whose product treatment was declined. It is still an ad
            # or a creator promo and must keep being counted as one; only the product
            # attachment is gone.
            if self.product_id:
                raise ValueError("treatment=none exposures must not carry a product_id.")
        elif not self.product_id:
            raise ValueError(f"source_type={self.source_type} requires a product_id for treatment={self.treatment}.")

    @property
    def is_commercial(self) -> bool:
        """True for ad/seller/affiliate supply, whether or not a product is attached."""
        return self.source_type in COMMERCIAL_SOURCE_TYPES

    @property
    def is_organic(self) -> bool:
        return self.source_type == "organic"

    @property
    def sells_a_product(self) -> bool:
        """True when this exposure actually puts a product in front of the viewer."""
        return self.product_id is not None and self.treatment != "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposure_id": self.exposure_id,
            "video_id": self.video_id,
            "source_type": self.source_type,
            "treatment": self.treatment,
            "placement": self.placement,
            "product_id": self.product_id,
            "base_scores": self.base_scores.to_dict(),
            "eligibility": self.eligibility.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExposureCandidate":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown ExposureCandidate fields: {sorted(unknown)}")
        data = dict(payload)
        base_scores = data.pop("base_scores", None)
        eligibility = data.pop("eligibility", None)
        return cls(
            **data,
            base_scores=BaseScores.from_dict(base_scores) if base_scores is not None else BaseScores(),
            eligibility=Eligibility.from_dict(eligibility) if eligibility is not None else Eligibility(),
        )


FEED_ACTION_TYPES = ("serve_exposure",)


@dataclass(frozen=True)
class FeedDecision:
    """Agent decision for the next exposure.

    v0.2 ships Top-1 ``serve_exposure`` only. ``rank_exposures`` is added once the
    Top-1 protocol is stable, which is why ``action_type`` is validated against an
    explicit allow-list rather than assumed.
    """

    action_type: str
    exposure_id: str
    reason: str = ""
    reasoning_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_type not in FEED_ACTION_TYPES:
            raise ValueError(
                f"Unknown feed action_type: {self.action_type}. Allowed: {list(FEED_ACTION_TYPES)}"
            )
        if not self.exposure_id:
            raise ValueError("serve_exposure requires an exposure_id.")

    @classmethod
    def serve(
        cls,
        exposure_id: str,
        reason: str = "",
        reasoning_summary: Mapping[str, Any] | None = None,
    ) -> "FeedDecision":
        return cls(
            action_type="serve_exposure",
            exposure_id=exposure_id,
            reason=reason,
            reasoning_summary=dict(reasoning_summary or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "exposure_id": self.exposure_id,
            "reason": self.reason,
            "reasoning_summary": copy.deepcopy(self.reasoning_summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeedDecision":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown FeedDecision fields: {sorted(unknown)}")
        data = dict(payload)
        return cls(
            action_type=str(data.get("action_type", "serve_exposure")),
            exposure_id=str(data.get("exposure_id", "")),
            reason=str(data.get("reason", "")),
            reasoning_summary=dict(data.get("reasoning_summary") or {}),
        )


FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "oracle",
        "gold",
        "gold_action",
        "gold_exposure_id",
        "gold_product_id",
        "expected_behaviors",
        "expected_action",
        "expected_actions",
        "scenario_type",
        "label",
        "labels",
        "answer",
        "solution",
        "hidden_world_state",
        "true_purchase_intent",
        "true_response_probabilities",
        "response_probabilities",
    }
)

LEAKY_TERMS = (
    "show_coupon",
    "delay_recommendation",
    "show_product_card",
    "switch_to_substitute",
    "show_explanation",
    "serve_exposure",
    "get_coupon",
    "find_substitute",
    "explain_recommendation",
    "rank_products",
    "retrieve_candidates",
    "submit_final_action",
    "gold",
    "expected behavior",
    "correct action",
)

_LEAKY_PHRASES = (
    r"use a valid coupon",
    r"show a coupon",
    r"avoid showing an unavailable coupon",
    r"delay recommendation",
    r"prefer a .{0,40}substitute",
    r"find a .{0,40}substitute",
    r"explain evidence before",
    r"you should (?:pick|choose|select|serve)",
    r"the (?:right|correct|best) (?:answer|choice|exposure)",
)


def find_forbidden_keys(payload: Any, _path: str = "") -> list[str]:
    """Recursively locate private keys anywhere in an agent-visible structure.

    A top-level-only check is not enough: ``{"user": {"true_purchase_intent": 0.9}}``
    is just as much of a leak as putting the key at the root.
    """

    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{_path}.{key}" if _path else str(key)
            if key in FORBIDDEN_PUBLIC_KEYS:
                hits.append(path)
            hits.extend(find_forbidden_keys(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            hits.extend(find_forbidden_keys(item, f"{_path}[{index}]"))
    return hits


def find_leaks(text: str) -> list[str]:
    """Return leakage markers found in agent-visible text.

    Guards the acceptance rule that an instruction must never name the action the
    agent is supposed to take.
    """

    if not text:
        return []
    lowered = text.lower()
    hits = [term for term in LEAKY_TERMS if term in lowered]
    for pattern in _LEAKY_PHRASES:
        match = re.search(pattern, lowered)
        if match:
            hits.append(match.group(0))
    return sorted(set(hits))


@dataclass
class ScenarioSpec:
    """A feed-control task with an explicit public/private boundary.

    Only ``scenario_id``, ``instruction``, ``public_context``, ``candidate_exposures``
    and the objective profile reach the agent. ``hidden_world_state`` drives the user
    simulator and ``oracle`` is evaluator-only.
    """

    scenario_id: str
    instruction: str = ""
    public_context: dict[str, Any] = field(default_factory=dict)
    hidden_world_state: dict[str, Any] = field(default_factory=dict)
    candidate_exposures: list[ExposureCandidate] = field(default_factory=list)
    oracle: dict[str, Any] = field(default_factory=dict)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    schema_version: str = FEED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("ScenarioSpec.scenario_id is required.")
        if not self.candidate_exposures:
            raise ValueError(f"Scenario {self.scenario_id} requires at least one candidate exposure.")

        exposure_ids = [candidate.exposure_id for candidate in self.candidate_exposures]
        duplicates = {value for value in exposure_ids if exposure_ids.count(value) > 1}
        if duplicates:
            raise ValueError(f"Scenario {self.scenario_id} has duplicate exposure_ids: {sorted(duplicates)}")

        forbidden = find_forbidden_keys(self.public_context)
        if forbidden:
            raise ValueError(
                f"Scenario {self.scenario_id} leaks private keys into public_context: {sorted(forbidden)}"
            )

        leaks = find_leaks(self.instruction)
        if leaks:
            raise ValueError(f"Scenario {self.scenario_id} instruction leaks the answer: {leaks}")

    def candidate(self, exposure_id: str) -> ExposureCandidate | None:
        for candidate in self.candidate_exposures:
            if candidate.exposure_id == exposure_id:
                return candidate
        return None

    def organic_candidates(self) -> list[ExposureCandidate]:
        return [candidate for candidate in self.candidate_exposures if candidate.is_organic]

    def commercial_candidates(self) -> list[ExposureCandidate]:
        return [candidate for candidate in self.candidate_exposures if candidate.is_commercial]

    def eligible_candidates(self) -> list[ExposureCandidate]:
        return [candidate for candidate in self.candidate_exposures if candidate.eligibility.is_eligible]

    def agent_view(self) -> dict[str, Any]:
        """Agent-visible projection. Never includes hidden_world_state or oracle."""
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "instruction": self.instruction,
            "public_context": copy.deepcopy(self.public_context),
            "candidate_exposures": [candidate.to_dict() for candidate in self.candidate_exposures],
            "objective_profile": self.reward_config.profile,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialization, including private state. Do not hand this to an agent."""
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "instruction": self.instruction,
            "public_context": copy.deepcopy(self.public_context),
            "hidden_world_state": copy.deepcopy(self.hidden_world_state),
            "candidate_exposures": [candidate.to_dict() for candidate in self.candidate_exposures],
            "oracle": copy.deepcopy(self.oracle),
            "reward_config": self.reward_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScenarioSpec":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown ScenarioSpec fields: {sorted(unknown)}")
        raw_candidates: Sequence[Mapping[str, Any]] = payload.get("candidate_exposures", [])
        raw_reward = payload.get("reward_config")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            instruction=str(payload.get("instruction", "")),
            public_context=dict(payload.get("public_context") or {}),
            hidden_world_state=dict(payload.get("hidden_world_state") or {}),
            candidate_exposures=[ExposureCandidate.from_dict(item) for item in raw_candidates],
            oracle=dict(payload.get("oracle") or {}),
            reward_config=RewardConfig.from_dict(raw_reward) if raw_reward is not None else RewardConfig(),
            schema_version=str(payload.get("schema_version", FEED_SCHEMA_VERSION)),
        )
