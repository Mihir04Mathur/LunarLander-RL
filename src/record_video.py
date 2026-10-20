"""Record a trained policy. Use matching seeds for comparable clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path("results") / "videos" / args.label
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(output), episode_trigger=lambda _: True)
    model = PPO.load(args.model)

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    print(f"Videos written to {output}")


if __name__ == "__main__":
    main()
