from __future__ import annotations

from videoshop.simulator.schemas import CommerceContext, EnvState, Product, SessionState, UserProfile, VideoContext


def build_mock_catalog() -> list[Product]:
    return [
        Product("P001", "Acrylic Desk Organizer", "home_organization", 24.99, 4.6, 430, 0.18, has_coupon=True, coupon_discount=0.15, tags=["minimal", "desk"]),
        Product("P002", "Cable Organizer Box", "home_organization", 15.99, 4.4, 210, 0.12, is_clearance=True, tags=["clean", "desk"]),
        Product("P003", "Portable Camping Lantern", "camping", 29.99, 4.7, 120, 0.10, is_high_margin=True, tags=["outdoor"]),
        Product("P004", "Makeup Brush Organizer", "beauty", 19.99, 4.5, 260, 0.22, tags=["aesthetic"]),
        Product("P005", "Low Cost Plastic Desk Organizer", "home_organization", 9.99, 4.1, 500, 0.28, tags=["desk"]),
        Product("P006", "Budget Camping Lantern", "camping", 18.99, 4.5, 180, 0.08, is_clearance=True, tags=["outdoor"]),
        Product("P007", "Viral Makeup Brush Organizer", "beauty", 22.99, 4.9, 95, 0.42, is_high_margin=True, tags=["aesthetic"]),
        Product("P008", "Aesthetic Brush Storage Tray", "home_organization", 16.99, 4.6, 150, 0.16, tags=["aesthetic", "desk"]),
    ]


def build_mock_videos() -> list[VideoContext]:
    return [
        VideoContext("V001", "Small desk makeover for tiny apartment", "home office desk setup", "home_organization", ["organizer", "lamp", "keyboard"], ["minimal", "clean"], "home lifestyle"),
        VideoContext("V002", "Weekend camping gear setup", "camping outdoor setup", "camping", ["lantern", "chair", "tent"], ["outdoor"], "travel creator"),
        VideoContext("V003", "Minimal vanity organization", "beauty desk organization", "beauty", ["brush", "mirror", "organizer"], ["aesthetic"], "beauty creator"),
    ]


def build_mock_users() -> list[UserProfile]:
    return [
        UserProfile("U001", "US", "medium", ["minimal", "clean"], {"home_organization": 0.70, "camping": 0.10, "beauty": 0.20}, 0.45, 0.35, 0.20, 0.35),
        UserProfile("U002", "US", "low", ["outdoor"], {"home_organization": 0.15, "camping": 0.75, "beauty": 0.10}, 0.80, 0.30, 0.35, 0.40),
        UserProfile("U003", "UK", "medium", ["aesthetic"], {"home_organization": 0.25, "camping": 0.10, "beauty": 0.70}, 0.55, 0.45, 0.25, 0.30),
        UserProfile("U004", "US", "low", ["aesthetic"], {"home_organization": 0.15, "camping": 0.10, "beauty": 0.75}, 0.85, 0.55, 0.18, 0.45),
    ]


def build_mock_commerce_context() -> CommerceContext:
    return CommerceContext(
        coupon_inventory={"P001": 12, "P003": 8},
        coupon_thresholds={"P001": 20.0, "P003": 25.0},
        coupon_expiry_steps={"P001": 5, "P003": 4},
        campaign_budget=5.0,
        stock_pressure={"P002": 0.8, "P005": 0.6},
        risk_constraints={"max_review_risk": 0.35},
    )


def build_mock_states() -> list[EnvState]:
    products = build_mock_catalog()
    videos = build_mock_videos()
    states: list[EnvState] = []
    for user in build_mock_users():
        states.append(
            EnvState(
                user_profile=user,
                session_state=SessionState(recent_watch_categories=[]),
                current_video=videos[0],
                candidate_products=products,
                commerce_context=build_mock_commerce_context(),
            )
        )
    return states
