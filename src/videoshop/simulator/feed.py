from __future__ import annotations

import random

from videoshop.simulator.schemas import UserProfile, VideoContext


class VideoFeed:
    def __init__(self, videos: list[VideoContext], rng: random.Random | None = None) -> None:
        if not videos:
            raise ValueError("VideoFeed requires at least one video.")
        self.videos = videos
        self.rng = rng or random.Random()

    def sample_next(self, user: UserProfile) -> VideoContext:
        scored = []
        for video in self.videos:
            interest = user.category_interests.get(video.category, 0.0)
            style_match = 0.1 * sum(style in user.style_preferences for style in video.styles)
            scored.append((interest + style_match + self.rng.random() * 0.05, video))
        return max(scored, key=lambda item: item[0])[1]

