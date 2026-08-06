from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoshop.env import VideoShopEnv
from videoshop.policies import RuleBasedPolicy
from videoshop.schemas import EnvState, Product, SessionState, UserProfile, VideoContext
from videoshop.trajectory import write_jsonl


def build_mock_states() -> list[EnvState]:
    products = [
        Product("P001", "亚克力桌面收纳盒", "家居收纳", 24.99, 4.6, 430, 0.18, tags=["极简", "桌面"]),
        Product("P002", "桌面理线收纳盒", "家居收纳", 15.99, 4.4, 210, 0.12, is_clearance=True, tags=["整洁", "桌面"]),
        Product("P003", "便携露营营地灯", "户外露营", 29.99, 4.7, 120, 0.10, is_high_margin=True, tags=["户外"]),
        Product("P004", "化妆刷桌面收纳架", "美妆收纳", 19.99, 4.5, 260, 0.22, tags=["高颜值"]),
    ]
    users = [
        UserProfile("U001", "美国", "中等预算", ["极简", "整洁"], ["家居收纳"], 0.45, 0.2, 0.35),
        UserProfile("U002", "美国", "低预算", ["户外"], ["户外露营"], 0.8, 0.35, 0.4),
        UserProfile("U003", "英国", "中等预算", ["高颜值"], ["美妆收纳"], 0.55, 0.25, 0.3),
    ]
    videos = [
        VideoContext("V001", "小户型桌面改造：把办公桌整理干净", "家居收纳 桌面布置", ["收纳", "台灯", "键盘"], ["极简", "整洁"], "家居生活创作者"),
        VideoContext("V002", "周末露营装备清单：轻量又实用", "户外露营 装备布置", ["营地灯", "折叠椅", "帐篷"], ["户外"], "旅行创作者"),
        VideoContext("V003", "极简化妆台收纳：刷具和镜子这样放", "美妆收纳 桌面整理", ["化妆刷", "镜子", "收纳"], ["高颜值"], "美妆创作者"),
    ]

    states: list[EnvState] = []
    for user in users:
        for video in videos:
            states.append(
                EnvState(
                    user_profile=user,
                    session_state=SessionState(step=0, recent_watch_categories=[video.scene.split()[0]]),
                    current_video=video,
                    candidate_products=products,
                )
            )
    return states


def generate(episodes: int, seed: int) -> tuple[list[dict], dict]:
    random.seed(seed)
    env = VideoShopEnv(build_mock_states(), max_steps=8, seed=seed)
    policy = RuleBasedPolicy()
    rows: list[dict] = []
    total_rewards: list[float] = []
    purchases = 0

    for episode_idx in range(episodes):
        state = env.reset()
        steps = []
        total_reward = 0.0
        outcome = "no_purchase"

        done = False
        while not done:
            action = policy.act(state)
            next_state, response, reward, done = env.step(action)
            total_reward += reward
            if response.purchased:
                purchases += 1
                outcome = "purchase"
            steps.append(
                {
                    "t": next_state.session_state.step,
                    "state": asdict(state),
                    "action": asdict(action),
                    "user_response": asdict(response),
                    "reward": reward,
                }
            )
            state = next_state

        total_rewards.append(total_reward)
        rows.append(
            {
                "episode_id": f"E{episode_idx + 1:06d}",
                "steps": steps,
                "total_reward": total_reward,
                "outcome": outcome,
            }
        )

    summary = {
        "episodes": episodes,
        "avg_reward": sum(total_rewards) / len(total_rewards),
        "purchase_rate": purchases / episodes,
        "policy": "RuleBasedPolicy",
        "language": "zh-CN",
        "说明": "JSON 字段名保持英文，轨迹中的场景、商品、用户画像和动作解释默认使用中文。",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/trajectories/mock_trajectories.jsonl")
    parser.add_argument("--report", default="outputs/reports/mock_eval_summary.json")
    args = parser.parse_args()

    rows, summary = generate(args.episodes, args.seed)
    write_jsonl(args.out, rows)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
