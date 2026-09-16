from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from videoshop.simulator.scenario import VideoShopScenario
from videoshop.simulator.schemas import CommerceContext, EnvState, Product, SessionState, UserProfile, VideoContext


@dataclass(frozen=True)
class SyntheticCommerceConfig:
    seed: int = 42
    category_count: int = 12
    products_per_category: int = 80
    users: int = 40
    videos_per_category: int = 8
    scenario_count: int = 50
    candidate_pool_size: int = 32
    coupon_coverage: float = 0.22
    clearance_rate: float = 0.12
    high_margin_rate: float = 0.18


_BASE_CATEGORIES = [
    ("home_organization", ["minimal", "clean", "desk"], ["organizer", "box", "rack", "tray"]),
    ("beauty", ["aesthetic", "vanity", "portable"], ["brush", "mirror", "storage", "palette"]),
    ("camping", ["outdoor", "rugged", "compact"], ["lantern", "chair", "tent", "stove"]),
    ("kitchen", ["modern", "compact", "easy_clean"], ["knife", "container", "pan", "scale"]),
    ("fitness", ["lightweight", "training", "home_gym"], ["band", "mat", "bottle", "roller"]),
    ("pet_supplies", ["durable", "washable", "cute"], ["feeder", "toy", "brush", "bed"]),
    ("electronics", ["portable", "smart", "desk"], ["charger", "hub", "stand", "cable"]),
    ("baby", ["soft", "safe", "washable"], ["bib", "mat", "bottle", "organizer"]),
    ("automotive", ["durable", "compact", "clean"], ["holder", "light", "cleaner", "pump"]),
    ("garden", ["outdoor", "durable", "compact"], ["planter", "tool", "sprayer", "light"]),
    ("travel", ["portable", "lightweight", "compact"], ["pouch", "bag", "adapter", "bottle"]),
    ("office", ["minimal", "ergonomic", "desk"], ["stand", "notebook", "lamp", "organizer"]),
]

_SYNTHETIC_SCENARIO_TYPES = [
    "coupon_valid",
    "fake_coupon_trap",
    "high_risk_explanation",
    "substitute_needed",
    "fatigue_delay",
    "low_inventory_guard",
    "stock_pressure_clearance",
    "category_mismatch_trap",
]


def build_synthetic_catalog(config: SyntheticCommerceConfig | None = None) -> list[Product]:
    config = config or SyntheticCommerceConfig()
    rng = random.Random(config.seed)
    products: list[Product] = []

    for category_index, (category, styles, objects) in enumerate(_category_specs(config.category_count)):
        for item_index in range(config.products_per_category):
            product_id = f"P{category_index + 1:03d}{item_index + 1:04d}"
            object_name = rng.choice(objects)
            style = rng.choice(styles)
            price = round(rng.uniform(6.99, 129.99), 2)
            rating = round(rng.uniform(3.7, 4.9), 1)
            review_risk = round(_bounded_normal(rng, mean=0.22, spread=0.14), 2)
            inventory = rng.randint(0, 2500)
            has_coupon = rng.random() < config.coupon_coverage
            coupon_discount = round(rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.20]), 2) if has_coupon else 0.0
            products.append(
                Product(
                    product_id=product_id,
                    title=f"{style.replace('_', ' ').title()} {category.replace('_', ' ').title()} {object_name.title()}",
                    category=category,
                    price=price,
                    rating=rating,
                    inventory=inventory,
                    review_risk=review_risk,
                    is_high_margin=rng.random() < config.high_margin_rate,
                    is_clearance=rng.random() < config.clearance_rate,
                    has_coupon=has_coupon,
                    coupon_discount=coupon_discount,
                    tags=[style, object_name],
                )
            )

    return products


def build_synthetic_videos(config: SyntheticCommerceConfig | None = None) -> list[VideoContext]:
    config = config or SyntheticCommerceConfig()
    videos: list[VideoContext] = []
    for category_index, (category, styles, objects) in enumerate(_category_specs(config.category_count)):
        for video_index in range(config.videos_per_category):
            style = styles[video_index % len(styles)]
            obj = objects[video_index % len(objects)]
            videos.append(
                VideoContext(
                    video_id=f"V{category_index + 1:03d}{video_index + 1:03d}",
                    caption=f"{style.replace('_', ' ')} {category.replace('_', ' ')} setup featuring {obj}",
                    scene=f"{category.replace('_', ' ')} demo with {obj}",
                    category=category,
                    objects=[obj],
                    styles=[style],
                    creator_type=f"{category} creator",
                )
            )
    return videos


def build_synthetic_users(config: SyntheticCommerceConfig | None = None) -> list[UserProfile]:
    config = config or SyntheticCommerceConfig()
    rng = random.Random(config.seed + 17)
    categories = [category for category, _, _ in _category_specs(config.category_count)]
    users: list[UserProfile] = []

    for index in range(config.users):
        primary = rng.choice(categories)
        secondary = rng.choice([category for category in categories if category != primary])
        interests = {category: round(rng.uniform(0.02, 0.18), 2) for category in categories}
        interests[primary] = round(rng.uniform(0.62, 0.92), 2)
        interests[secondary] = round(rng.uniform(0.25, 0.55), 2)
        style_preferences = _styles_for_categories([primary, secondary], config.category_count)
        users.append(
            UserProfile(
                user_id=f"U{index + 1:05d}",
                country=rng.choice(["US", "UK", "DE", "JP", "SG"]),
                budget_level=rng.choice(["low", "medium", "high"]),
                style_preferences=rng.sample(style_preferences, k=min(3, len(style_preferences))),
                category_interests=interests,
                price_sensitivity=round(rng.uniform(0.15, 0.95), 2),
                risk_sensitivity=round(rng.uniform(0.20, 0.85), 2),
                ad_fatigue=round(rng.uniform(0.05, 0.65), 2),
                purchase_intent=round(rng.uniform(0.15, 0.70), 2),
            )
        )
    return users


def build_synthetic_commerce_context(
    products: list[Product],
    config: SyntheticCommerceConfig | None = None,
) -> CommerceContext:
    config = config or SyntheticCommerceConfig()
    rng = random.Random(config.seed + 31)
    coupon_products = [product for product in products if product.has_coupon]
    clearance_products = [product for product in products if product.is_clearance]
    return CommerceContext(
        coupon_inventory={product.product_id: rng.randint(0, 120) for product in coupon_products},
        coupon_thresholds={product.product_id: round(max(0.0, product.price * rng.uniform(0.45, 0.95)), 2) for product in coupon_products},
        coupon_expiry_steps={product.product_id: rng.randint(-2, 14) for product in coupon_products},
        campaign_budget=round(max(10.0, len(coupon_products) * 0.35), 2),
        stock_pressure={product.product_id: round(rng.uniform(0.35, 0.98), 2) for product in clearance_products},
        risk_constraints={"max_review_risk": 0.35},
    )


def build_synthetic_states(config: SyntheticCommerceConfig | None = None) -> list[EnvState]:
    config = config or SyntheticCommerceConfig()
    catalog = build_synthetic_catalog(config)
    videos = build_synthetic_videos(config)
    users = build_synthetic_users(config)
    context = build_synthetic_commerce_context(catalog, config)
    rng = random.Random(config.seed + 47)
    states: list[EnvState] = []

    for user in users:
        video = rng.choice(videos)
        candidates = _sample_candidate_pool(catalog, video, config.candidate_pool_size, rng)
        states.append(
            EnvState(
                user_profile=copy.deepcopy(user),
                session_state=SessionState(),
                current_video=copy.deepcopy(video),
                candidate_products=copy.deepcopy(candidates),
                commerce_context=_slice_commerce_context(context, candidates),
            )
        )
    return states


def build_synthetic_scenarios(config: SyntheticCommerceConfig | None = None) -> list[VideoShopScenario]:
    config = config or SyntheticCommerceConfig()
    catalog = build_synthetic_catalog(config)
    videos = build_synthetic_videos(config)
    users = build_synthetic_users(config)
    context = build_synthetic_commerce_context(catalog, config)
    rng = random.Random(config.seed + 53)
    scenarios: list[VideoShopScenario] = []

    for index in range(config.scenario_count):
        scenario_type = _SYNTHETIC_SCENARIO_TYPES[index % len(_SYNTHETIC_SCENARIO_TYPES)]
        video = rng.choice(videos)
        user = copy.deepcopy(rng.choice(users))
        _boost_interest(user, video.category)
        candidates = copy.deepcopy(_sample_candidate_pool(catalog, video, config.candidate_pool_size, rng))
        scenario_context = _slice_commerce_context(context, candidates)
        session_state = _synthetic_session_state(rng)
        objective, expected_behaviors = _specialize_synthetic_scenario(
            scenario_type,
            user,
            video,
            candidates,
            scenario_context,
            session_state,
            rng,
        )
        scenarios.append(
            VideoShopScenario(
                scenario_id=f"synthetic_{scenario_type}_{index + 1:04d}",
                user_profile=user,
                initial_session_state=session_state,
                video_feed=[copy.deepcopy(video)],
                candidate_products=candidates,
                commerce_context=scenario_context,
                objective=objective,
                expected_behaviors=expected_behaviors,
            )
        )
    return scenarios


def _category_specs(category_count: int) -> list[tuple[str, list[str], list[str]]]:
    specs = list(_BASE_CATEGORIES)
    for index in range(len(specs), category_count):
        specs.append(
            (
                f"category_{index + 1:03d}",
                [f"style_{index + 1}_a", f"style_{index + 1}_b", "value"],
                [f"object_{index + 1}_a", f"object_{index + 1}_b", "kit"],
            )
        )
    return specs[:category_count]


def _styles_for_categories(categories: list[str], category_count: int) -> list[str]:
    styles: list[str] = []
    for category, category_styles, _ in _category_specs(category_count):
        if category in categories:
            styles.extend(category_styles)
    return sorted(set(styles))


def _sample_candidate_pool(
    catalog: list[Product],
    video: VideoContext,
    pool_size: int,
    rng: random.Random,
) -> list[Product]:
    relevant = [product for product in catalog if product.category == video.category]
    distractors = [product for product in catalog if product.category != video.category]
    relevant_count = max(4, min(len(relevant), pool_size // 2))
    distractor_count = max(0, pool_size - relevant_count)
    candidates = rng.sample(relevant, k=min(relevant_count, len(relevant)))
    candidates.extend(rng.sample(distractors, k=min(distractor_count, len(distractors))))
    rng.shuffle(candidates)
    return candidates


def _slice_commerce_context(context: CommerceContext, products: list[Product]) -> CommerceContext:
    product_ids = {product.product_id for product in products}
    return CommerceContext(
        coupon_inventory={key: value for key, value in context.coupon_inventory.items() if key in product_ids},
        coupon_thresholds={key: value for key, value in context.coupon_thresholds.items() if key in product_ids},
        coupon_expiry_steps={key: value for key, value in context.coupon_expiry_steps.items() if key in product_ids},
        campaign_budget=context.campaign_budget,
        stock_pressure={key: value for key, value in context.stock_pressure.items() if key in product_ids},
        risk_constraints=dict(context.risk_constraints),
    )


def _boost_interest(user: UserProfile, category: str) -> None:
    user.category_interests[category] = max(user.category_interests.get(category, 0.0), 0.65)


def _specialize_synthetic_scenario(
    scenario_type: str,
    user: UserProfile,
    video: VideoContext,
    candidates: list[Product],
    context: CommerceContext,
    session_state: SessionState,
    rng: random.Random,
) -> tuple[str, list[str]]:
    anchor = _anchor_product(candidates, video)
    _align_product_to_video(anchor, video)

    if scenario_type == "coupon_valid":
        user.price_sensitivity = max(user.price_sensitivity, 0.75)
        _make_coupon_available(anchor, context, rng)
        return (
            f"Use a valid coupon for a relevant {video.category} product when coupon inventory and budget allow it.",
            ["call_get_coupon_before_show_coupon", "avoid_fake_coupon", "prefer_relevant_coupon"],
        )

    if scenario_type == "fake_coupon_trap":
        user.price_sensitivity = max(user.price_sensitivity, 0.80)
        _make_coupon_unavailable(anchor, context)
        return (
            f"Avoid showing an unavailable coupon for a relevant {video.category} product.",
            ["call_get_coupon_before_show_coupon", "avoid_fake_coupon", "fallback_to_product_card_or_substitute"],
        )

    if scenario_type == "high_risk_explanation":
        anchor.review_risk = 0.52
        anchor.rating = max(anchor.rating, 4.7)
        anchor.inventory = max(anchor.inventory, 20)
        context.risk_constraints["max_review_risk"] = 0.35
        return (
            f"Explain evidence before recommending a relevant but high-risk {video.category} product.",
            ["call_explain_recommendation", "avoid_unexplained_high_risk"],
        )

    if scenario_type == "substitute_needed":
        user.price_sensitivity = max(user.price_sensitivity, 0.85)
        anchor.price = max(anchor.price, 79.99)
        anchor.review_risk = max(anchor.review_risk, 0.32)
        _make_coupon_unavailable(anchor, context)
        substitute = _ensure_substitute(candidates, anchor, video, rng)
        context.stock_pressure[substitute.product_id] = max(context.stock_pressure.get(substitute.product_id, 0.0), 0.7)
        return (
            f"Find a cheaper, lower-risk substitute when the leading {video.category} product is expensive or unsupported.",
            ["call_find_substitute", "prefer_lower_price_substitute", "prefer_lower_risk_substitute"],
        )

    if scenario_type == "fatigue_delay":
        user.ad_fatigue = 0.92
        session_state.recent_skips = max(session_state.recent_skips, 3)
        return (
            "Delay recommendations when ad fatigue and recent skips are high.",
            ["delay_recommendation", "avoid_interrupting_fatigued_user"],
        )

    if scenario_type == "low_inventory_guard":
        anchor.inventory = 0
        substitute = _ensure_substitute(candidates, anchor, video, rng)
        substitute.inventory = max(substitute.inventory, 80)
        return (
            f"Avoid an out-of-stock {video.category} product and prefer an in-stock substitute.",
            ["avoid_out_of_stock", "call_find_substitute", "prefer_available_product"],
        )

    if scenario_type == "stock_pressure_clearance":
        user.price_sensitivity = max(user.price_sensitivity, 0.78)
        clearance = _ensure_substitute(candidates, anchor, video, rng)
        clearance.is_clearance = True
        clearance.price = min(clearance.price, max(6.99, anchor.price * 0.75))
        context.stock_pressure[clearance.product_id] = 0.95
        return (
            f"Prefer a relevant clearance substitute when stock pressure and user price sensitivity are high.",
            ["call_find_substitute", "prefer_clearance_substitute", "avoid_unavailable_coupon"],
        )

    if scenario_type == "category_mismatch_trap":
        distractor = _category_mismatch_distractor(candidates, video, rng)
        distractor.rating = 4.9
        _make_coupon_available(distractor, context, rng)
        return (
            f"Avoid a tempting coupon product whose category mismatches the current {video.category} video.",
            ["avoid_category_mismatch", "prefer_video_category_alignment", "avoid_irrelevant_coupon"],
        )

    return (
        _objective_for_synthetic_scenario(video.category),
        ["ground_actions_in_tools", "avoid_unavailable_coupon", "avoid_category_mismatch", "prefer_in_stock_low_risk_products"],
    )


def _synthetic_session_state(rng: random.Random) -> SessionState:
    return SessionState(recent_skips=rng.choice([0, 0, 1, 2, 3]))


def _objective_for_synthetic_scenario(category: str) -> str:
    return (
        f"Assist a realistic short-video commerce session for category={category}. "
        "Use tools for coupon, ranking, substitute, and explanation evidence before taking risky actions."
    )


def _bounded_normal(rng: random.Random, mean: float, spread: float) -> float:
    return max(0.0, min(0.85, rng.gauss(mean, spread)))


def _anchor_product(candidates: list[Product], video: VideoContext) -> Product:
    for product in candidates:
        if product.category == video.category:
            return product
    return candidates[0]


def _align_product_to_video(product: Product, video: VideoContext) -> None:
    product.category = video.category
    for tag in [*video.styles, *video.objects]:
        if tag not in product.tags:
            product.tags.append(tag)
    if video.objects and video.objects[0] not in product.title.lower():
        product.title = f"{product.title} {video.objects[0].title()}"
    product.inventory = max(product.inventory, 1)
    product.review_risk = min(product.review_risk, 0.34)


def _make_coupon_available(product: Product, context: CommerceContext, rng: random.Random) -> None:
    product.has_coupon = True
    product.coupon_discount = product.coupon_discount or rng.choice([0.08, 0.10, 0.12, 0.15, 0.20])
    context.coupon_inventory[product.product_id] = max(context.coupon_inventory.get(product.product_id, 0), rng.randint(8, 80))
    context.coupon_thresholds[product.product_id] = min(product.price, round(product.price * 0.65, 2))
    context.coupon_expiry_steps[product.product_id] = max(context.coupon_expiry_steps.get(product.product_id, 0), rng.randint(4, 12))
    context.campaign_budget = max(context.campaign_budget, product.coupon_discount * 10)


def _make_coupon_unavailable(product: Product, context: CommerceContext) -> None:
    product.has_coupon = True
    product.coupon_discount = product.coupon_discount or 0.15
    context.coupon_inventory[product.product_id] = 0
    context.coupon_thresholds[product.product_id] = min(product.price, round(product.price * 0.65, 2))
    context.coupon_expiry_steps[product.product_id] = -1


def _ensure_substitute(
    candidates: list[Product],
    anchor: Product,
    video: VideoContext,
    rng: random.Random,
) -> Product:
    options = [
        product
        for product in candidates
        if product.product_id != anchor.product_id and product.category == video.category
    ]
    substitute = options[0] if options else candidates[-1]
    substitute.category = video.category
    substitute.price = round(max(4.99, min(substitute.price, anchor.price * rng.uniform(0.45, 0.82))), 2)
    substitute.review_risk = round(max(0.02, min(anchor.review_risk, 0.22)), 2)
    substitute.inventory = max(substitute.inventory, 50)
    substitute.rating = max(substitute.rating, 4.3)
    substitute.is_clearance = True
    for tag in [*video.styles, *video.objects]:
        if tag not in substitute.tags:
            substitute.tags.append(tag)
    return substitute


def _category_mismatch_distractor(
    candidates: list[Product],
    video: VideoContext,
    rng: random.Random,
) -> Product:
    mismatches = [product for product in candidates if product.category != video.category]
    if mismatches:
        return rng.choice(mismatches)
    product = candidates[-1]
    product.category = f"mismatch_{video.category}"
    return product
