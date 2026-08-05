from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserProfile:
    user_id: str
    country: str
    budget_level: str
    style_preferences: list[str]
    category_interests: list[str]
    price_sensitivity: float
    ad_fatigue: float
    purchase_intent: float


@dataclass
class VideoContext:
    video_id: str
    caption: str
    scene: str
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
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    step: int
    recent_watch_categories: list[str] = field(default_factory=list)
    recent_clicks: list[str] = field(default_factory=list)
    recent_carts: list[str] = field(default_factory=list)
    recent_skips: int = 0


@dataclass
class EnvState:
    user_profile: UserProfile
    session_state: SessionState
    current_video: VideoContext
    candidate_products: list[Product]


@dataclass
class AgentAction:
    action_type: str
    product_id: str | None = None
    product_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class UserResponse:
    clicked: bool = False
    increased_watch_time: bool = False
    added_to_cart: bool = False
    purchased: bool = False
    bundle_purchased: bool = False
    ad_clicked: bool = False
    returned_or_refunded: bool = False
    irrelevant_recommendation: bool = False
    interrupted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

