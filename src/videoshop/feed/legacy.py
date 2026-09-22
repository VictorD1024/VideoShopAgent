from __future__ import annotations

from typing import Any

from videoshop.feed.schemas import FEED_SCHEMA_VERSION, BaseScores, Eligibility, ExposureCandidate
from videoshop.simulator.trajectory import SCHEMA_VERSION_V1, schema_version_of

# v1 intervention actions mapped onto v2 exposure treatments.
LEGACY_ACTION_TO_TREATMENT = {
    "delay_recommendation": "none",
    "show_product_card": "product_anchor",
    "show_coupon": "coupon",
    "switch_to_substitute": "product_anchor",
    "show_explanation": "product_anchor",
}


def treatment_for_legacy_action(action_type: str) -> str:
    try:
        return LEGACY_ACTION_TO_TREATMENT[action_type]
    except KeyError:
        raise ValueError(
            f"Unknown legacy action_type: {action_type}. Known: {sorted(LEGACY_ACTION_TO_TREATMENT)}"
        ) from None


def legacy_action_of(step: dict[str, Any]) -> dict[str, Any]:
    """Extract the final action from either v1 step layout.

    Rollout and benchmark episodes store it under ``action``; LLM tool-calling
    episodes store it under ``agent_step.final_action`` and have no ``action`` key
    at all. Defaulting to an empty action here would silently rewrite every
    commercial decision as an organic no-op, so an unrecognized step is an error.
    """

    action = step.get("action")
    if isinstance(action, dict) and action.get("action_type"):
        return action

    final_action = (step.get("agent_step") or {}).get("final_action")
    if isinstance(final_action, dict) and final_action.get("action_type"):
        return final_action

    raise ValueError(
        "Legacy step carries no final action under 'action' or 'agent_step.final_action'; "
        f"available keys: {sorted(step)}"
    )


def exposure_from_legacy_step(
    step: dict[str, Any],
    *,
    exposure_id: str | None = None,
    warnings: list[str] | None = None,
) -> ExposureCandidate:
    """Rebuild the exposure a v1 step implicitly served.

    v1 had no feed layer: the environment fixed ``current_video`` and the agent only
    chose a product treatment. The reconstructed candidate is therefore always the
    one the environment had already committed to.
    """

    action = legacy_action_of(step)
    video = _legacy_video(step)
    video_id = video.get("video_id", "legacy_video")
    action_type = action.get("action_type", "delay_recommendation")
    treatment = treatment_for_legacy_action(action_type)
    product_id = action.get("product_id")

    if treatment != "none" and not product_id:
        # Real v1 logs contain product actions with a null product_id. Downgrading is
        # the only representable outcome, but it must not pass unrecorded.
        if warnings is not None:
            warnings.append(f"product_action_without_product_id:{action_type}")
        treatment = "none"
        product_id = None
    if not video_id:
        if warnings is not None:
            warnings.append("missing_video_id")
        video_id = "legacy_video"

    source_type = "seller" if product_id else "organic"

    return ExposureCandidate(
        exposure_id=exposure_id or f"legacy:{video_id}:{step.get('t', 0)}",
        video_id=video_id,
        source_type=source_type,
        treatment=treatment,
        placement="for_you",
        product_id=product_id,
        base_scores=BaseScores(),
        eligibility=Eligibility(),
    )


def upgrade_v1_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Project a v1 intervention episode into the v2 envelope.

    Reward stays the v1 scalar under ``legacy_reward``: v1 never recorded the
    component breakdown, so synthesizing a RewardVector here would invent data.
    """

    version = schema_version_of(episode)
    if version == FEED_SCHEMA_VERSION:
        return dict(episode)
    if version != SCHEMA_VERSION_V1:
        raise ValueError(f"Cannot upgrade unknown schema_version: {version}")

    steps = []
    episode_warnings: list[str] = []
    for step in episode.get("steps", []):
        action = legacy_action_of(step)
        warnings: list[str] = []
        candidate = exposure_from_legacy_step(step, warnings=warnings)
        episode_warnings.extend(warnings)
        steps.append(
            {
                "schema_version": FEED_SCHEMA_VERSION,
                "t": step.get("t"),
                "candidate_exposure_ids": [candidate.exposure_id],
                "decision": {
                    "action_type": "serve_exposure",
                    "exposure_id": candidate.exposure_id,
                    "reason": action.get("reason", ""),
                    "reasoning_summary": dict(action.get("reasoning_summary") or {}),
                },
                "served_exposure": candidate.to_dict(),
                "blocked": False,
                "block_reasons": [],
                "user_response": step.get("user_response"),
                "reward_vector": None,
                "reward_terms": {},
                "legacy_reward": step.get("reward"),
                "legacy_action": action,
                "legacy_tool_calls": _legacy_tool_calls(step),
                "conversion_warnings": warnings,
            }
        )

    return {
        "schema_version": FEED_SCHEMA_VERSION,
        "converted_from": SCHEMA_VERSION_V1,
        "episode_id": episode.get("episode_id"),
        "scenario_id": episode.get("scenario_id"),
        "policy": episode.get("policy", "legacy"),
        "objective_profile": None,
        "steps": steps,
        "total_reward_vector": None,
        "legacy_total_reward": episode.get("total_reward"),
        "legacy_outcome": episode.get("outcome"),
        "conversion_warnings": episode_warnings,
    }


def is_feed_episode(episode: dict[str, Any]) -> bool:
    return schema_version_of(episode) == FEED_SCHEMA_VERSION


def _legacy_video(step: dict[str, Any]) -> dict[str, Any]:
    state = step.get("state") or {}
    video = state.get("current_video")
    if isinstance(video, dict):
        return video
    # LLM episodes also snapshot the agent-visible observation.
    observation = step.get("observation") or {}
    return observation.get("current_video") or {}


def _legacy_tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool evidence, from whichever v1 layout recorded it."""
    action = step.get("action")
    if isinstance(action, dict) and action.get("tool_calls"):
        return list(action["tool_calls"])
    return [
        {"tool": result.get("tool"), "input": result.get("input"), "output": result.get("output")}
        for result in step.get("tool_results") or []
        if result.get("success", True)
    ]
