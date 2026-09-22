from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # Import for typing only; schemas imports RewardConfig from here.
    from videoshop.feed.candidate_provider import ExposureTruth
    from videoshop.feed.schemas import ExposureCandidate
    from videoshop.feed.user_model import FeedUserResponse
    from videoshop.simulator.schemas import UserProfile

REWARD_COMPONENTS = (
    "content_value",
    "commerce_value",
    "user_value",
    "ecosystem_value",
    "risk_cost",
)


@dataclass(frozen=True)
class RewardVector:
    """Auditable multi-objective reward for one feed decision.

    The first four components are values (higher is better). ``risk_cost`` is a
    non-negative magnitude that scalarization subtracts, so a raw vector never
    hides a penalty inside a positive-looking component.
    """

    content_value: float = 0.0
    commerce_value: float = 0.0
    user_value: float = 0.0
    ecosystem_value: float = 0.0
    risk_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in REWARD_COMPONENTS:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"RewardVector.{name} must be a number, got {type(value).__name__}")
            object.__setattr__(self, name, float(value))
        if self.risk_cost < 0.0:
            raise ValueError("RewardVector.risk_cost must be non-negative; encode benefits as value components.")

    def __add__(self, other: "RewardVector") -> "RewardVector":
        if not isinstance(other, RewardVector):
            return NotImplemented
        return RewardVector(**{name: getattr(self, name) + getattr(other, name) for name in REWARD_COMPONENTS})

    def scalarize(self, weights: "RewardWeights | str | Mapping[str, float] | None" = None) -> float:
        resolved = resolve_weights(weights)
        return (
            resolved.content_value * self.content_value
            + resolved.commerce_value * self.commerce_value
            + resolved.user_value * self.user_value
            + resolved.ecosystem_value * self.ecosystem_value
            - resolved.risk_cost * self.risk_cost
        )

    def contributions(self, weights: "RewardWeights | str | Mapping[str, float] | None" = None) -> dict[str, float]:
        """Per-component signed contribution to the scalar, for audit trails."""
        resolved = resolve_weights(weights)
        signed = {
            name: getattr(resolved, name) * getattr(self, name)
            for name in REWARD_COMPONENTS
            if name != "risk_cost"
        }
        signed["risk_cost"] = -resolved.risk_cost * self.risk_cost
        return signed

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in REWARD_COMPONENTS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RewardVector":
        unknown = set(payload) - set(REWARD_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown RewardVector fields: {sorted(unknown)}")
        return cls(**{name: float(payload[name]) for name in REWARD_COMPONENTS if name in payload})

    @classmethod
    def zero(cls) -> "RewardVector":
        return cls()


@dataclass(frozen=True)
class RewardWeights:
    """Non-negative scalarization weights over :class:`RewardVector`."""

    content_value: float = 1.0
    commerce_value: float = 1.0
    user_value: float = 1.0
    ecosystem_value: float = 1.0
    risk_cost: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"RewardWeights.{field.name} must be a number, got {type(value).__name__}")
            if value < 0.0:
                raise ValueError(f"RewardWeights.{field.name} must be non-negative; flip the component sign instead.")
            object.__setattr__(self, field.name, float(value))

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in REWARD_COMPONENTS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RewardWeights":
        unknown = set(payload) - set(REWARD_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown RewardWeights fields: {sorted(unknown)}")
        return cls(**{name: float(value) for name, value in payload.items()})


SCALARIZATIONS: dict[str, RewardWeights] = {
    "content_first": RewardWeights(
        content_value=1.0,
        commerce_value=0.25,
        user_value=1.0,
        ecosystem_value=0.5,
        risk_cost=1.0,
    ),
    "balanced": RewardWeights(
        content_value=1.0,
        commerce_value=1.0,
        user_value=1.0,
        ecosystem_value=0.5,
        risk_cost=1.0,
    ),
    "gmv_first": RewardWeights(
        content_value=0.3,
        commerce_value=1.6,
        user_value=0.5,
        ecosystem_value=0.3,
        risk_cost=0.8,
    ),
    "retention_first": RewardWeights(
        content_value=1.2,
        commerce_value=0.4,
        user_value=1.6,
        ecosystem_value=0.6,
        risk_cost=1.4,
    ),
    "clearance_campaign": RewardWeights(
        content_value=0.5,
        commerce_value=1.0,
        user_value=0.7,
        ecosystem_value=1.6,
        risk_cost=1.0,
    ),
}

DEFAULT_SCALARIZATION = "balanced"


def resolve_weights(weights: RewardWeights | str | Mapping[str, float] | None) -> RewardWeights:
    if weights is None:
        return SCALARIZATIONS[DEFAULT_SCALARIZATION]
    if isinstance(weights, RewardWeights):
        return weights
    if isinstance(weights, str):
        try:
            return SCALARIZATIONS[weights]
        except KeyError:
            raise ValueError(
                f"Unknown scalarization profile: {weights}. Available: {sorted(SCALARIZATIONS)}"
            ) from None
    return RewardWeights.from_dict(weights)


@dataclass(frozen=True)
class RewardConfig:
    """Objective profile attached to a scenario.

    ``profile`` is public (it is the platform objective the agent optimizes for);
    it carries no information about which candidate is correct.
    """

    profile: str = DEFAULT_SCALARIZATION
    weights: RewardWeights | None = None

    def __post_init__(self) -> None:
        if self.weights is None and self.profile not in SCALARIZATIONS:
            raise ValueError(
                f"Unknown scalarization profile: {self.profile}. Available: {sorted(SCALARIZATIONS)}"
            )

    def resolved_weights(self) -> RewardWeights:
        return self.weights if self.weights is not None else SCALARIZATIONS[self.profile]

    def scalarize(self, vector: RewardVector) -> float:
        return vector.scalarize(self.resolved_weights())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"profile": self.profile}
        if self.weights is not None:
            payload["weights"] = self.weights.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RewardConfig":
        unknown = set(payload) - {"profile", "weights"}
        if unknown:
            raise ValueError(f"Unknown RewardConfig fields: {sorted(unknown)}")
        raw_weights = payload.get("weights")
        return cls(
            profile=str(payload.get("profile", DEFAULT_SCALARIZATION)),
            weights=RewardWeights.from_dict(raw_weights) if raw_weights is not None else None,
        )


@dataclass(frozen=True)
class FeedRewardConfig:
    """Coefficients turning one realized exposure outcome into a reward vector."""

    watch_seconds_scale: float = 20.0
    watch_credit_cap: float = 1.5
    skip_penalty: float = 0.6
    organic_relevance: float = 0.4

    margin_scale: float = 10.0
    click_funnel_credit: float = 0.15
    cart_funnel_credit: float = 0.25

    relevance_satisfaction: float = 0.5
    click_satisfaction: float = 0.2
    annoyance_penalty: float = 0.8
    session_exit_penalty: float = 1.5
    wasted_commercial_penalty: float = 0.3

    affiliate_supply_credit: float = 0.4
    seller_supply_credit: float = 0.3
    clearance_relief_credit: float = 0.6
    wasted_ad_supply_penalty: float = 0.2

    refund_cost: float = 1.0
    risk_exposure_cost: float = 0.5
    annoyance_risk_cost: float = 0.3
    fatigue_risk_cost: float = 0.6
    fatigue_threshold: float = 0.7

    invalid_decision_cost: float = 1.0
    intervention_violation_cost: float = 1.2
    """Cost per ungrounded intervention (fake coupon, unsupported explanation, ...).

    Carries the v1 grounding penalties into the v2 reward vector so that composing
    the two layers does not quietly drop them.
    """


DEFAULT_FEED_REWARD_CONFIG = FeedRewardConfig()


def reward_terms(
    candidate: "ExposureCandidate",
    truth: "ExposureTruth",
    response: "FeedUserResponse",
    user: "UserProfile",
    config: FeedRewardConfig | None = None,
    violations: "Sequence[str]" = (),
) -> dict[str, dict[str, float]]:
    """Per-component, per-term reward breakdown for audit trails."""

    cfg = config or DEFAULT_FEED_REWARD_CONFIG
    watched = response.watched_seconds
    commercial = candidate.source_type != "organic"
    net_purchase = response.purchased and not response.refunded

    content: dict[str, float] = {
        "watch_credit": min(cfg.watch_credit_cap, watched / cfg.watch_seconds_scale),
    }
    if response.skipped:
        content["skip_penalty"] = -cfg.skip_penalty
    else:
        content["relevance_credit"] = cfg.organic_relevance * truth.category_match

    commerce: dict[str, float] = {}
    if net_purchase:
        commerce["net_margin"] = truth.price * truth.margin_rate / cfg.margin_scale
    if response.clicked:
        commerce["click_credit"] = cfg.click_funnel_credit
    if response.added_to_cart:
        commerce["cart_credit"] = cfg.cart_funnel_credit

    user_value: dict[str, float] = {}
    if not response.skipped:
        user_value["satisfaction"] = cfg.relevance_satisfaction * truth.category_match
    if response.clicked:
        user_value["intent_served"] = cfg.click_satisfaction
    if response.annoyed:
        user_value["annoyance"] = -cfg.annoyance_penalty
    if response.left_session:
        user_value["session_exit"] = -cfg.session_exit_penalty
    if commercial and response.skipped:
        user_value["wasted_interruption"] = -cfg.wasted_commercial_penalty

    ecosystem: dict[str, float] = {}
    if candidate.source_type == "affiliate" and not response.skipped:
        ecosystem["creator_supply"] = cfg.affiliate_supply_credit
    if candidate.source_type == "seller" and response.clicked:
        ecosystem["seller_supply"] = cfg.seller_supply_credit
    if net_purchase and truth.clearance_relief:
        ecosystem["clearance_relief"] = cfg.clearance_relief_credit * truth.clearance_relief
    if candidate.source_type == "ad" and response.skipped:
        ecosystem["wasted_ad_supply"] = -cfg.wasted_ad_supply_penalty

    risk: dict[str, float] = {}
    if response.refunded:
        risk["refund"] = cfg.refund_cost
    if response.clicked and truth.refund_probability:
        risk["risk_exposure"] = cfg.risk_exposure_cost * truth.refund_probability
    if response.annoyed:
        risk["annoyance"] = cfg.annoyance_risk_cost
    if user.ad_fatigue >= cfg.fatigue_threshold and commercial:
        risk["fatigue_pressure"] = cfg.fatigue_risk_cost
    for violation in violations:
        risk[f"intervention:{violation}"] = cfg.intervention_violation_cost

    return {
        "content_value": content,
        "commerce_value": commerce,
        "user_value": user_value,
        "ecosystem_value": ecosystem,
        "risk_cost": risk,
    }


def compute_reward_vector(
    candidate: "ExposureCandidate",
    truth: "ExposureTruth",
    response: "FeedUserResponse",
    user: "UserProfile",
    config: FeedRewardConfig | None = None,
    violations: "Sequence[str]" = (),
) -> RewardVector:
    terms = reward_terms(candidate, truth, response, user, config, violations)
    return RewardVector(
        **{component: round(sum(values.values()), 6) for component, values in terms.items()}
    )


def expected_reward_vector(
    candidate: "ExposureCandidate",
    truth: "ExposureTruth",
    user: "UserProfile",
    recent_skips: int = 0,
    config: FeedRewardConfig | None = None,
) -> RewardVector:
    """Full-information expectation of one exposure, used only by the evaluator oracle.

    This is the analytic counterpart of :func:`compute_reward_vector`; it consumes the
    private truth directly and must never be reachable from an agent observation.
    """

    cfg = config or DEFAULT_FEED_REWARD_CONFIG
    skip = truth.skip_probability
    watch = truth.watch_seconds * ((1.0 - skip) + 0.18 * skip)
    click = truth.click_probability * (1.0 - skip)
    purchase = min(click, truth.purchase_probability)
    net_purchase = purchase * (1.0 - truth.refund_probability)
    cart = purchase + (click - purchase) * 0.45
    commercial = candidate.source_type != "organic"
    exit_probability = min(1.0, 0.015 + 0.30 * truth.annoyance + 0.10 * skip + 0.05 * recent_skips)

    content = (
        min(cfg.watch_credit_cap, watch / cfg.watch_seconds_scale)
        - cfg.skip_penalty * skip
        + cfg.organic_relevance * truth.category_match * (1.0 - skip)
    )
    commerce = (
        truth.price * truth.margin_rate / cfg.margin_scale * net_purchase
        + cfg.click_funnel_credit * click
        + cfg.cart_funnel_credit * cart
    )
    user_value = (
        cfg.relevance_satisfaction * truth.category_match * (1.0 - skip)
        + cfg.click_satisfaction * click
        - cfg.annoyance_penalty * truth.annoyance
        - cfg.session_exit_penalty * exit_probability
        - (cfg.wasted_commercial_penalty * skip if commercial else 0.0)
    )
    ecosystem = 0.0
    if candidate.source_type == "affiliate":
        ecosystem += cfg.affiliate_supply_credit * (1.0 - skip)
    if candidate.source_type == "seller":
        ecosystem += cfg.seller_supply_credit * click
    if truth.clearance_relief:
        ecosystem += cfg.clearance_relief_credit * truth.clearance_relief * net_purchase
    if candidate.source_type == "ad":
        ecosystem -= cfg.wasted_ad_supply_penalty * skip

    risk = (
        cfg.refund_cost * purchase * truth.refund_probability
        + cfg.risk_exposure_cost * truth.refund_probability * click
        + cfg.annoyance_risk_cost * truth.annoyance
    )
    if user.ad_fatigue >= cfg.fatigue_threshold and commercial:
        risk += cfg.fatigue_risk_cost

    return RewardVector(
        content_value=round(content, 6),
        commerce_value=round(commerce, 6),
        user_value=round(user_value, 6),
        ecosystem_value=round(ecosystem, 6),
        risk_cost=round(max(0.0, risk), 6),
    )
