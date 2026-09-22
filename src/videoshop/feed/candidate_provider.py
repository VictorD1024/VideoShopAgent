from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from videoshop.feed.schemas import BaseScores, Eligibility, ExposureCandidate
from videoshop.simulator.schemas import Product, SessionState, UserProfile, VideoContext

SOURCE_MIX = ("organic", "seller", "affiliate", "ad")

# Per-source interruption pressure. Organic content never annoys on its own.
_SOURCE_ANNOYANCE = {"organic": 0.0, "seller": 0.10, "affiliate": 0.08, "ad": 0.22}
# Watch-time handicap of commercial formats relative to organic content.
_SOURCE_WATCH_PENALTY = {"organic": 0.0, "seller": 0.12, "affiliate": 0.08, "ad": 0.25}


@dataclass(frozen=True)
class CandidateProviderConfig:
    candidates_per_step: int = 8
    min_organic: int = 2
    min_commercial: int = 3
    ineligible_rate: float = 0.22
    coupon_treatment_rate: float = 0.35
    coupon_trap_rate: float = 0.3
    """Share of coupon exposures whose advertised coupon will not actually redeem."""
    score_noise: float = 0.10
    commercial_optimism: float = 0.35
    """How much the underlying ranker over-predicts commercial purchase intent.

    This is the core imperfection: a policy that trusts base_scores and always
    serves the highest predicted GMV will over-expose commerce and lose user value.
    """

    def __post_init__(self) -> None:
        if self.candidates_per_step < self.min_organic + self.min_commercial:
            raise ValueError(
                "candidates_per_step must cover min_organic + min_commercial so that every step "
                "offers both organic and commercial choices."
            )
        if not 0.0 <= self.ineligible_rate <= 0.9:
            raise ValueError("ineligible_rate must be within [0, 0.9].")


@dataclass(frozen=True)
class ExposureTruth:
    """Ground-truth response parameters. Environment-private, never shown to agents."""

    watch_seconds: float
    skip_probability: float
    click_probability: float
    purchase_probability: float
    refund_probability: float
    annoyance: float
    price: float = 0.0
    margin_rate: float = 0.0
    clearance_relief: float = 0.0
    category_match: float = 0.0
    coupon_available: bool = True
    """Whether the advertised coupon actually redeems.

    The candidate's ``eligibility.coupon_valid`` is the *ranker's* belief and can be
    stale. Only ``get_coupon`` in the intervention layer settles it, so a feed-only
    policy cannot tell a real offer from a trap.
    """

    def to_dict(self) -> dict[str, float]:
        return {
            "watch_seconds": self.watch_seconds,
            "skip_probability": self.skip_probability,
            "click_probability": self.click_probability,
            "purchase_probability": self.purchase_probability,
            "refund_probability": self.refund_probability,
            "annoyance": self.annoyance,
            "price": self.price,
            "margin_rate": self.margin_rate,
            "clearance_relief": self.clearance_relief,
            "category_match": self.category_match,
            "coupon_available": self.coupon_available,
        }


@dataclass
class CandidateBatch:
    """One step of feed candidates plus the private truth behind them."""

    candidates: list[ExposureCandidate] = field(default_factory=list)
    truth: dict[str, ExposureTruth] = field(default_factory=dict)
    videos: dict[str, VideoContext] = field(default_factory=dict)
    products: dict[str, Product] = field(default_factory=dict)

    def public_catalog(self) -> dict[str, dict]:
        """Agent-visible descriptions of the videos and products in this batch."""
        return {
            "videos": {
                video_id: {
                    "caption": video.caption,
                    "scene": video.scene,
                    "category": video.category,
                    "objects": list(video.objects),
                    "styles": list(video.styles),
                    "creator_type": video.creator_type,
                }
                for video_id, video in self.videos.items()
            },
            "products": {
                product_id: {
                    "title": product.title,
                    "category": product.category,
                    "price": product.price,
                    "rating": product.rating,
                    "inventory": product.inventory,
                    "review_risk": product.review_risk,
                    "has_coupon": product.has_coupon,
                    "coupon_discount": product.coupon_discount,
                    "is_clearance": product.is_clearance,
                    "tags": list(product.tags),
                }
                for product_id, product in self.products.items()
            },
        }


class SyntheticCandidateProvider:
    """Stands in for the underlying recommender's shortlist.

    It returns a bounded, mixed candidate set per step: organic clips, seller
    livestream-style clips, affiliate creator clips and paid ads. Its ``base_scores``
    are a deliberately noisy and commerce-optimistic view of ``ExposureTruth``.
    """

    def __init__(
        self,
        products: list[Product],
        videos: list[VideoContext],
        config: CandidateProviderConfig | None = None,
    ) -> None:
        if not products:
            raise ValueError("SyntheticCandidateProvider requires a non-empty product catalog.")
        if not videos:
            raise ValueError("SyntheticCandidateProvider requires a non-empty video pool.")
        self.products = products
        self.videos = videos
        self.config = config or CandidateProviderConfig()
        self._videos_by_category: dict[str, list[VideoContext]] = {}
        for video in videos:
            self._videos_by_category.setdefault(video.category, []).append(video)
        self._products_by_category: dict[str, list[Product]] = {}
        for product in products:
            self._products_by_category.setdefault(product.category, []).append(product)

    def generate(
        self,
        user: UserProfile,
        session: SessionState,
        step: int,
        rng: random.Random,
        *,
        exposure_prefix: str = "X",
        placement: str = "for_you",
    ) -> CandidateBatch:
        config = self.config
        plan = self._source_plan(rng)
        batch = CandidateBatch()

        for index, source_type in enumerate(plan):
            video = self._pick_video(user, rng, prefer_interest=source_type == "organic")
            product = None if source_type == "organic" else self._pick_product(video, rng)
            if source_type != "organic" and product is None:
                source_type = "organic"

            treatment = "none"
            if product is not None:
                treatment = "coupon" if product.has_coupon and rng.random() < config.coupon_treatment_rate else "product_anchor"

            exposure_id = f"{exposure_prefix}{step:02d}_{index:02d}"
            # What the ranker advertises, and therefore what base_scores describe.
            advertised = self.truth_for(user, session, video, product, source_type, treatment)
            coupon_available = not (
                treatment == "coupon" and rng.random() < config.coupon_trap_rate
            )
            if coupon_available:
                truth = replace(advertised, coupon_available=True)
            else:
                # A coupon that will not redeem buys nothing: no conversion lift and no
                # discounted price. Both environments must agree on that; only the
                # intervention layer can additionally catch the false claim.
                truth = replace(
                    self.truth_for(user, session, video, product, source_type, "product_anchor"),
                    coupon_available=False,
                )
            candidate = ExposureCandidate(
                exposure_id=exposure_id,
                video_id=video.video_id,
                source_type=source_type,
                treatment=treatment,
                placement=placement,
                product_id=product.product_id if product else None,
                base_scores=self._noisy_scores(advertised, source_type, product, rng),
                eligibility=self._eligibility(product, treatment, source_type, rng),
            )

            batch.candidates.append(candidate)
            batch.truth[exposure_id] = truth
            batch.videos[video.video_id] = video
            if product is not None:
                batch.products[product.product_id] = product

        self._guarantee_servable_organic(batch, user, session, rng, step, exposure_prefix, placement)
        return batch

    def _source_plan(self, rng: random.Random) -> list[str]:
        config = self.config
        plan = ["organic"] * config.min_organic
        plan.extend(["seller", "affiliate", "ad"][: config.min_commercial])
        while len(plan) < config.min_commercial + config.min_organic:
            plan.append(rng.choice(("seller", "affiliate", "ad")))
        while len(plan) < config.candidates_per_step:
            plan.append(rng.choice(SOURCE_MIX))
        rng.shuffle(plan)
        return plan

    def _pick_video(self, user: UserProfile, rng: random.Random, *, prefer_interest: bool) -> VideoContext:
        categories = list(self._videos_by_category)
        if prefer_interest and rng.random() < 0.7:
            weights = [max(0.01, user.category_interests.get(category, 0.0)) for category in categories]
            category = rng.choices(categories, weights=weights, k=1)[0]
        else:
            category = rng.choice(categories)
        return rng.choice(self._videos_by_category[category])

    def _pick_product(self, video: VideoContext, rng: random.Random) -> Product | None:
        pool = self._products_by_category.get(video.category)
        if not pool:
            return None
        return rng.choice(pool)

    def truth_for(
        self,
        user: UserProfile,
        session: SessionState,
        video: VideoContext,
        product: Product | None,
        source_type: str,
        treatment: str,
    ) -> ExposureTruth:
        """Private response parameters for one `video x product x treatment` combination.

        Public so the composite environment can re-derive truth after the intervention
        layer changes the treatment, but never reachable from an agent observation.
        """
        interest = user.category_interests.get(video.category, 0.0)
        style_match = 0.08 * sum(style in user.style_preferences for style in video.styles)
        affinity = _clamp(interest + style_match)
        fatigue = _clamp(user.ad_fatigue + 0.04 * session.recent_skips)

        watch = max(1.5, 4.0 + 16.0 * affinity) * (1.0 - _SOURCE_WATCH_PENALTY[source_type])
        skip = _clamp(0.62 - 0.50 * affinity + 0.35 * fatigue * _SOURCE_ANNOYANCE[source_type] / 0.22)
        annoyance = _clamp(_SOURCE_ANNOYANCE[source_type] * (0.6 + fatigue))

        if product is None:
            return ExposureTruth(
                watch_seconds=watch,
                skip_probability=skip,
                click_probability=0.0,
                purchase_probability=0.0,
                refund_probability=0.0,
                annoyance=annoyance,
                category_match=affinity,
            )

        coupon_boost = 0.06 if treatment == "coupon" else 0.0
        price_drag = _clamp(user.price_sensitivity * min(1.0, product.price / 120.0))
        rating_lift = 0.06 if product.rating >= 4.5 else 0.0
        # Risk-averse viewers inspect reviews and back off; risk-tolerant ones do not.
        risk_drag = _clamp(user.risk_sensitivity * product.review_risk)

        click = _clamp(0.03 + 0.55 * affinity + coupon_boost + rating_lift - 0.18 * fatigue - 0.30 * risk_drag)
        purchase = _clamp(
            click
            * _clamp(0.22 + 0.70 * user.purchase_intent - 0.40 * price_drag + 1.5 * coupon_boost - 0.60 * risk_drag)
        )
        refund = _clamp(0.15 * product.review_risk + 0.55 * product.review_risk * (1.0 - affinity))

        return ExposureTruth(
            watch_seconds=watch,
            skip_probability=skip,
            click_probability=click,
            purchase_probability=purchase,
            refund_probability=refund,
            annoyance=annoyance,
            price=product.price * (1.0 - (product.coupon_discount if treatment == "coupon" else 0.0)),
            margin_rate=0.28 if product.is_high_margin else 0.14,
            clearance_relief=1.0 if product.is_clearance else 0.0,
            category_match=affinity,
        )

    def _noisy_scores(
        self,
        truth: ExposureTruth,
        source_type: str,
        product: Product | None,
        rng: random.Random,
    ) -> BaseScores:
        noise = self.config.score_noise
        optimism = 1.0 + (self.config.commercial_optimism if source_type != "organic" else 0.0)

        click = _jitter(truth.click_probability * optimism, noise, rng)
        purchase = _jitter(truth.purchase_probability * optimism, noise, rng)
        refund = _jitter(truth.refund_probability * 0.7, noise, rng)  # the ranker under-predicts refunds
        price = truth.price if product else 0.0

        return BaseScores(
            expected_watch_time=max(0.0, _jitter_raw(truth.watch_seconds, noise * 6.0, rng)),
            skip_probability=_jitter(truth.skip_probability, noise, rng),
            product_click_probability=click,
            purchase_probability=min(click, purchase),
            expected_net_gmv=round(max(0.0, price * purchase * (1.0 - refund)), 4),
            refund_probability=refund,
        )

    def _eligibility(
        self,
        product: Product | None,
        treatment: str,
        source_type: str,
        rng: random.Random,
    ) -> Eligibility:
        if product is None:
            # Organic clips can still be pulled for policy reasons, but far more rarely.
            return Eligibility(policy_compliant=rng.random() >= self.config.ineligible_rate * 0.25)

        roll = rng.random()
        in_stock = product.inventory > 0 and roll >= self.config.ineligible_rate * 0.45
        coupon_valid = treatment != "coupon" or rng.random() >= self.config.ineligible_rate * 0.6
        risk_within_limit = product.review_risk < 0.55
        policy_compliant = rng.random() >= self.config.ineligible_rate * 0.2
        audience_allowed = source_type != "ad" or rng.random() >= self.config.ineligible_rate * 0.2

        return Eligibility(
            in_stock=in_stock,
            coupon_valid=coupon_valid,
            policy_compliant=policy_compliant,
            risk_within_limit=risk_within_limit,
            audience_allowed=audience_allowed,
        )

    def _guarantee_servable_organic(
        self,
        batch: CandidateBatch,
        user: UserProfile,
        session: SessionState,
        rng: random.Random,
        step: int,
        exposure_prefix: str,
        placement: str,
    ) -> None:
        """Doing nothing commercial must always be a legal move.

        Without this, "no intervention" could become unavailable and the benchmark
        would silently force commerce.
        """

        servable_organic = [
            candidate
            for candidate in batch.candidates
            if candidate.is_organic and candidate.eligibility.is_eligible
        ]
        if servable_organic:
            return

        video = self._pick_video(user, rng, prefer_interest=True)
        exposure_id = f"{exposure_prefix}{step:02d}_organic"
        truth = self.truth_for(user, session, video, None, "organic", "none")
        batch.candidates.append(
            ExposureCandidate(
                exposure_id=exposure_id,
                video_id=video.video_id,
                source_type="organic",
                placement=placement,
                base_scores=self._noisy_scores(truth, "organic", None, rng),
                eligibility=Eligibility(),
            )
        )
        batch.truth[exposure_id] = truth
        batch.videos[video.video_id] = video


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _jitter(value: float, noise: float, rng: random.Random) -> float:
    return _clamp(round(rng.gauss(value, noise), 4))


def _jitter_raw(value: float, noise: float, rng: random.Random) -> float:
    return round(rng.gauss(value, noise), 4)
