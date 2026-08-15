from videoshop.data.mock import build_mock_states, build_mock_videos
from videoshop.policies import RuleBasedPolicy
from videoshop.simulator.env import VideoShopEnv
from videoshop.simulator.rollout import run_episode


def test_run_episode_records_pre_action_state_snapshot():
    env = VideoShopEnv(build_mock_states(), build_mock_videos(), max_steps=2, seed=42)
    episode = run_episode(env, RuleBasedPolicy(), "E000001")

    first_step = episode["steps"][0]
    second_step = episode["steps"][1]

    assert first_step["t"] == 1
    assert first_step["state"]["session_state"]["step"] == 0
    assert second_step["t"] == 2
    assert second_step["state"]["session_state"]["step"] == 1
