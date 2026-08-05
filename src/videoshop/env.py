from __future__ import annotations

import random

from videoshop.reward import compute_reward
from videoshop.schemas import AgentAction, EnvState, UserResponse
from videoshop.user_simulator import UserSimulator, find_product


class VideoShopEnv:
    def __init__(self, initial_states: list[EnvState], max_steps: int = 8, seed: int = 42) -> None:
        self.initial_states = initial_states
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.user_simulator = UserSimulator(self.rng)
        self.state: EnvState | None = None

    def reset(self) -> EnvState:
        self.state = self.rng.choice(self.initial_states)
        self.state.session_state.step = 0
        self.state.session_state.recent_clicks.clear()
        self.state.session_state.recent_carts.clear()
        self.state.session_state.recent_skips = 0
        return self.state

    def step(self, action: AgentAction) -> tuple[EnvState, UserResponse, float, bool]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        response = self.user_simulator.respond(self.state, action)
        product = find_product(self.state, action.product_id)
        reward = compute_reward(response, product)

        self.state.session_state.step += 1
        if product and response.clicked:
            self.state.session_state.recent_clicks.append(product.product_id)
        if product and response.added_to_cart:
            self.state.session_state.recent_carts.append(product.product_id)
        if response.interrupted or response.irrelevant_recommendation:
            self.state.session_state.recent_skips += 1

        done = self.state.session_state.step >= self.max_steps or response.purchased
        return self.state, response, reward, done

