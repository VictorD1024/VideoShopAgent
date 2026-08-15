from __future__ import annotations

import copy
import random

from videoshop.simulator.feed import VideoFeed
from videoshop.simulator.reward import compute_reward
from videoshop.simulator.schemas import AgentAction, EnvState, Product, StateUpdate, UserResponse, VideoContext
from videoshop.simulator.state import apply_state_update
from videoshop.simulator.tools import product_match_score
from videoshop.simulator.user_simulator import UserSimulator


class VideoShopEnv:
    def __init__(
        self,
        initial_states: list[EnvState],
        videos: list[VideoContext],
        max_steps: int = 8,
        seed: int = 42,
    ) -> None:
        if not initial_states:
            raise ValueError("VideoShopEnv requires at least one initial state.")
        self.initial_states = initial_states
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.feed = VideoFeed(videos, self.rng)
        self.user_simulator = UserSimulator(self.rng)
        self.state: EnvState | None = None

    def reset(self) -> EnvState:
        self.state = copy.deepcopy(self.rng.choice(self.initial_states))
        self.state.session_state.step = 0
        self.state.current_video = self.feed.sample_next(self.state.user_profile)
        return self.state

    def step(self, action: AgentAction) -> tuple[EnvState, UserResponse, float, bool, StateUpdate]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        product = self.find_product(action.product_id)
        response = self.user_simulator.respond(self.state, action, product)
        category_match = product is None or product_match_score(self.state, product) >= 0.25
        reward = compute_reward(response, action, product, category_match=category_match)
        update = apply_state_update(self.state, action, response, product)
        done = update.episode_done or self.state.session_state.step >= self.max_steps

        if not done:
            self.state.current_video = self.feed.sample_next(self.state.user_profile)
            self.state.session_state.recent_watch_categories.append(self.state.current_video.category)

        return self.state, response, reward, done, update

    def find_product(self, product_id: str | None) -> Product | None:
        if self.state is None or product_id is None:
            return None
        for product in self.state.candidate_products:
            if product.product_id == product_id:
                return product
        return None
