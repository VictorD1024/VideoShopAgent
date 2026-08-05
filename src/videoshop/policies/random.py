from __future__ import annotations

import random

from videoshop.simulator.schemas import AgentAction, EnvState


class RandomPolicy:
    action_types = [
        "delay_recommendation",
        "show_product_card",
        "show_coupon",
        "switch_to_substitute",
        "show_explanation",
    ]

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def act(self, state: EnvState) -> AgentAction:
        action_type = self.rng.choice(self.action_types)
        product = self.rng.choice(state.candidate_products) if state.candidate_products else None
        return AgentAction(
            action_type=action_type,
            product_id=product.product_id if product else None,
            reason="Random baseline action.",
        )

