from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from videoshop.feed.schemas import ExposureCandidate
from videoshop.simulator.observation import ALLOWED_ACTIONS, ALLOWED_TOOLS, build_observation
from videoshop.simulator.schemas import EnvState

COUPON_AUTHORITY_FIELDS = ("coupon_inventory", "coupon_thresholds", "coupon_expiry_steps")
"""Commerce signals that decide whether a coupon redeems.

`coupon_is_available` is a pure function of these plus public product and session
fields, so leaving them in the prompt lets a model settle the question by arithmetic
and never call `get_coupon`. In the composite environment the tool is the only
authority, which means these have to come out of the agent's view.
"""


@dataclass
class InterventionObservation:
    """What an intervention agent sees in the composite environment.

    Same shape as the v1 `Observation`, minus the coupon authority fields, plus the
    exposure the feed layer already committed to. The v1 observation is left untouched
    so frozen v1 trajectories stay reproducible.
    """

    task: str
    exposure: dict[str, Any]
    user_summary: dict[str, Any] = field(default_factory=dict)
    session_summary: dict[str, Any] = field(default_factory=dict)
    current_video: dict[str, Any] = field(default_factory=dict)
    visible_candidates: list[dict[str, Any]] = field(default_factory=list)
    commerce_signals: dict[str, Any] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=lambda: list(ALLOWED_TOOLS))
    allowed_actions: list[str] = field(default_factory=lambda: list(ALLOWED_ACTIONS))
    last_step: dict[str, Any] | None = None


INTERVENTION_TASK = (
    "The feed has already committed to this clip and its anchor product. Decide how to "
    "present it. The offered treatment is a proposal from the ranker, not a fact: verify "
    "a coupon with get_coupon before showing it, and only leave the anchor product "
    "through find_substitute."
)


def build_intervention_observation(
    state: EnvState,
    exposure: ExposureCandidate,
    *,
    task: str = INTERVENTION_TASK,
    last_step: dict[str, Any] | None = None,
    candidate_limit: int | None = None,
) -> InterventionObservation:
    base = build_observation(state, task=task, last_step=last_step, candidate_limit=candidate_limit)
    commerce_signals = {
        key: value
        for key, value in base.commerce_signals.items()
        if key not in COUPON_AUTHORITY_FIELDS
    }

    return InterventionObservation(
        task=base.task,
        exposure={
            "exposure_id": exposure.exposure_id,
            "video_id": exposure.video_id,
            "source_type": exposure.source_type,
            "placement": exposure.placement,
            "anchor_product_id": exposure.product_id,
            "offered_treatment": exposure.treatment,
            # The ranker's claim about the offer, which may be stale.
            "offered_eligibility": exposure.eligibility.to_dict(),
        },
        user_summary=base.user_summary,
        session_summary=base.session_summary,
        current_video=base.current_video,
        visible_candidates=base.visible_candidates,
        commerce_signals=commerce_signals,
        allowed_tools=list(base.allowed_tools),
        allowed_actions=list(base.allowed_actions),
        last_step=base.last_step,
    )


def build_intervention_system_prompt() -> str:
    return (
        "You are the intervention layer of a short-video commerce feed.\n"
        "The feed layer has already chosen the clip and, for commercial supply, an anchor "
        "product. You cannot change the clip, and you may only leave the anchor product by "
        "calling find_substitute and then switching to exactly the product it returns.\n"
        "\n"
        "observation.exposure tells you the content source (ad, seller, affiliate or "
        "organic), the anchor product, and the treatment the ranker proposes.\n"
        "\n"
        "Rules:\n"
        "- offered_treatment is a proposal. A proposed coupon may be stale.\n"
        "- Call get_coupon before show_coupon. If it reports unavailable, do not claim it.\n"
        "- Coupon availability is not in the observation. Only get_coupon is authoritative.\n"
        "- Evidence must match the tool result and name the product you finally choose.\n"
        "- The clip airs either way, so declining the product does not remove the ad; "
        "decline only when the anchor is genuinely unsellable.\n"
        "Finish every turn by calling submit_final_action."
    )


def intervention_few_shot_messages() -> list[dict[str, str]]:
    """Few-shots that match the composite protocol, not the v1 one.

    The v1 examples teach the model to pick a product from the video. Here the product
    is already chosen; the examples therefore start from `observation.exposure` and
    never imply that coupon inventory is sitting in the prompt.
    """

    return [
        {
            "role": "user",
            "content": (
                '{"observation":{"exposure":{"anchor_product_id":"P001",'
                '"offered_treatment":"coupon","source_type":"ad"}}}'
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Call get_coupon for P001. The offered coupon is a claim, not a fact. "
                "If available, call submit_final_action with action_type=show_coupon "
                "for P001. If unavailable, call submit_final_action with "
                "action_type=show_product_card for P001. Include reasoning_summary."
            ),
        },
        {
            "role": "user",
            "content": (
                '{"observation":{"exposure":{"anchor_product_id":"P002",'
                '"offered_treatment":"product_anchor","source_type":"seller"},'
                '"visible_candidates":[{"product_id":"P002","inventory":0,'
                '"review_risk":0.7}]}}'
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Call find_substitute for P002. If it returns a product, call "
                "submit_final_action with action_type=switch_to_substitute for exactly "
                "that product_id. If it returns none, delay. Include reasoning_summary."
            ),
        },
    ]
