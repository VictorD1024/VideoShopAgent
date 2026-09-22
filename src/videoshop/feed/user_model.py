from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from videoshop.feed.candidate_provider import ExposureTruth
from videoshop.feed.schemas import ExposureCandidate
from videoshop.simulator.schemas import SessionState, UserProfile


@dataclass
class FeedUserResponse:
    """Observed outcome of one served exposure."""

    watched_seconds: float = 0.0
    skipped: bool = False
    clicked: bool = False
    added_to_cart: bool = False
    purchased: bool = False
    refunded: bool = False
    annoyed: bool = False
    left_session: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "watched_seconds": self.watched_seconds,
            "skipped": self.skipped,
            "clicked": self.clicked,
            "added_to_cart": self.added_to_cart,
            "purchased": self.purchased,
            "refunded": self.refunded,
            "annoyed": self.annoyed,
            "left_session": self.left_session,
            "metadata": dict(self.metadata),
        }


@dataclass
class FeedStateUpdate:
    ad_fatigue_delta: float = 0.0
    purchase_intent_delta: float = 0.0
    interest_delta: dict[str, float] = field(default_factory=dict)
    session_ended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ad_fatigue_delta": round(self.ad_fatigue_delta, 6),
            "purchase_intent_delta": round(self.purchase_intent_delta, 6),
            "interest_delta": {key: round(value, 6) for key, value in self.interest_delta.items()},
            "session_ended": self.session_ended,
        }


class FeedUserModel:
    """Samples user behaviour from private ``ExposureTruth``.

    Agents only ever see the provider's noisy ``base_scores``, so a policy cannot
    read these probabilities off the observation.
    """

    def respond(
        self,
        candidate: ExposureCandidate,
        truth: ExposureTruth,
        user: UserProfile,
        session: SessionState,
        rng: random.Random,
    ) -> FeedUserResponse:
        skipped = rng.random() < truth.skip_probability
        watched = truth.watch_seconds * (0.18 if skipped else 1.0)

        clicked = False
        purchased = False
        added_to_cart = False
        refunded = False

        if not skipped and truth.click_probability > 0.0:
            clicked = rng.random() < truth.click_probability
            if clicked:
                conditional_purchase = min(1.0, truth.purchase_probability / max(truth.click_probability, 1e-6))
                purchased = rng.random() < conditional_purchase
                added_to_cart = purchased or rng.random() < 0.45
                refunded = purchased and rng.random() < truth.refund_probability

        annoyed = rng.random() < truth.annoyance
        exit_probability = _clamp(
            0.015 + 0.30 * truth.annoyance + 0.10 * float(skipped) + 0.05 * session.recent_skips
        )
        left_session = rng.random() < exit_probability

        return FeedUserResponse(
            watched_seconds=round(watched, 3),
            skipped=skipped,
            clicked=clicked,
            added_to_cart=added_to_cart,
            purchased=purchased,
            refunded=refunded,
            annoyed=annoyed,
            left_session=left_session,
            metadata={
                "source_type": candidate.source_type,
                "treatment": candidate.treatment,
                "category_match": round(truth.category_match, 4),
            },
        )

    def apply(
        self,
        response: FeedUserResponse,
        candidate: ExposureCandidate,
        truth: ExposureTruth,
        user: UserProfile,
        session: SessionState,
        video_category: str,
    ) -> FeedStateUpdate:
        update = FeedStateUpdate()
        session.step += 1
        session.recent_watch_categories.append(video_category)

        if candidate.is_commercial:
            update.ad_fatigue_delta += 0.05 + 0.10 * float(response.annoyed)
        else:
            update.ad_fatigue_delta -= 0.04 if not response.skipped else 0.0

        if response.skipped:
            session.recent_skips += 1
            update.ad_fatigue_delta += 0.03
        else:
            session.recent_skips = max(0, session.recent_skips - 1)
            update.interest_delta[video_category] = 0.03

        if candidate.product_id:
            session.exposed_products.append(candidate.product_id)
        if response.clicked and candidate.product_id:
            session.recent_clicks.append(candidate.product_id)
            update.purchase_intent_delta += 0.08
            update.interest_delta[video_category] = update.interest_delta.get(video_category, 0.0) + 0.06
        if response.added_to_cart and candidate.product_id:
            session.recent_carts.append(candidate.product_id)
            update.purchase_intent_delta += 0.12
        if (
            candidate.treatment == "coupon"
            and candidate.product_id
            and response.clicked
            and truth.coupon_available
        ):
            # A stale coupon burns no allowance, so the public session must not claim it
            # was used. Otherwise the trap leaks into the next step's observation.
            session.used_coupons.append(candidate.product_id)
        if response.purchased:
            update.purchase_intent_delta -= 0.25
        if response.refunded:
            update.purchase_intent_delta -= 0.15
            update.ad_fatigue_delta += 0.05

        update.session_ended = response.left_session

        user.ad_fatigue = _clamp(user.ad_fatigue + update.ad_fatigue_delta)
        user.purchase_intent = _clamp(user.purchase_intent + update.purchase_intent_delta)
        for category, delta in update.interest_delta.items():
            user.category_interests[category] = _clamp(user.category_interests.get(category, 0.0) + delta)

        return update


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
