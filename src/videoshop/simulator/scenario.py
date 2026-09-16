from __future__ import annotations

import copy
from dataclasses import dataclass, field

from videoshop.simulator.schemas import CommerceContext, EnvState, Product, SessionState, UserProfile, VideoContext


@dataclass
class VideoShopScenario:
    scenario_id: str
    user_profile: UserProfile
    initial_session_state: SessionState
    video_feed: list[VideoContext]
    candidate_products: list[Product]
    commerce_context: CommerceContext
    objective: str = "Optimize short-video commerce assistance without unsupported claims."
    constraints: dict[str, object] = field(default_factory=dict)
    expected_behaviors: list[str] = field(default_factory=list)

    @classmethod
    def from_state(
        cls,
        scenario_id: str,
        state: EnvState,
        video_feed: list[VideoContext],
        objective: str = "Optimize short-video commerce assistance without unsupported claims.",
    ) -> "VideoShopScenario":
        return cls(
            scenario_id=scenario_id,
            user_profile=copy.deepcopy(state.user_profile),
            initial_session_state=copy.deepcopy(state.session_state),
            video_feed=copy.deepcopy(video_feed),
            candidate_products=copy.deepcopy(state.candidate_products),
            commerce_context=copy.deepcopy(state.commerce_context),
            objective=objective,
        )

    def to_initial_state(self) -> EnvState:
        if not self.video_feed:
            raise ValueError(f"Scenario {self.scenario_id} requires at least one video.")
        return EnvState(
            user_profile=copy.deepcopy(self.user_profile),
            session_state=copy.deepcopy(self.initial_session_state),
            current_video=copy.deepcopy(self.video_feed[0]),
            candidate_products=copy.deepcopy(self.candidate_products),
            commerce_context=copy.deepcopy(self.commerce_context),
        )
