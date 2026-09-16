from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserProfile:
    user_id: str
    country: str
    budget_level: str
    style_preferences: list[str]
    category_interests: dict[str, float]
    price_sensitivity: float
    risk_sensitivity: float
    ad_fatigue: float
    purchase_intent: float


@dataclass
class VideoContext:
    video_id: str
    caption: str
    scene: str
    category: str
    objects: list[str]
    styles: list[str]
    creator_type: str


@dataclass
class Product:
    product_id: str
    title: str
    category: str
    price: float
    rating: float
    inventory: int
    review_risk: float
    is_high_margin: bool = False
    is_clearance: bool = False
    has_coupon: bool = False
    coupon_discount: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    step: int = 0
    recent_watch_categories: list[str] = field(default_factory=list)
    recent_clicks: list[str] = field(default_factory=list)
    recent_carts: list[str] = field(default_factory=list)
    recent_skips: int = 0
    exposed_products: list[str] = field(default_factory=list)
    used_coupons: list[str] = field(default_factory=list)


@dataclass
class CommerceContext:
    coupon_inventory: dict[str, int] = field(default_factory=dict)
    coupon_thresholds: dict[str, float] = field(default_factory=dict)
    coupon_expiry_steps: dict[str, int] = field(default_factory=dict)
    campaign_budget: float = 0.0
    stock_pressure: dict[str, float] = field(default_factory=dict)
    risk_constraints: dict[str, float] = field(default_factory=lambda: {"max_review_risk": 0.35})


@dataclass
class EnvState:
    user_profile: UserProfile
    session_state: SessionState
    current_video: VideoContext
    candidate_products: list[Product]
    commerce_context: CommerceContext = field(default_factory=CommerceContext)


@dataclass
class ToolCall:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class AgentAction:
    action_type: str
    product_id: str | None = None
    reason: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserResponse:
    clicked: bool = False
    increased_watch_time: bool = False
    added_to_cart: bool = False
    purchased: bool = False
    skipped: bool = False
    returned_or_refunded: bool = False
    irrelevant_recommendation: bool = False
    interrupted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateUpdate:
    purchase_intent_delta: float = 0.0
    ad_fatigue_delta: float = 0.0
    interest_updates: dict[str, float] = field(default_factory=dict)
    episode_done: bool = False


@dataclass
class Observation:
    task: str
    user_summary: dict[str, Any]
    session_summary: dict[str, Any]
    current_video: dict[str, Any]
    visible_candidates: list[dict[str, Any]]
    commerce_signals: dict[str, Any]
    allowed_tools: list[str]
    allowed_actions: list[str]
    last_step: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolCallRequest:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class FinalAction:
    action_type: str
    product_id: str | None = None
    reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    reasoning_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    tool_requests: list[ToolCallRequest] = field(default_factory=list)
    final_action: FinalAction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
