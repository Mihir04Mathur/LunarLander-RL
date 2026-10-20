"""Record a polished, post-training Lunar Lander video with telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

from src.cinematic import CinematicRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PPO model path, with or without .zip")
    parser.add_argument("--label", required=True, help="Folder name for the rendered videos")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def terminal_label(native_return: float, terminated: bool) -> str:
    if native_return >= 200:
        return "MISSION SUCCESS - LANDED"
    if terminated:
        return "MISSION ENDED - CRASH OR ABORT"
    return "TIME LIMIT REACHED"


def main() -> None:
    args = parse_args()
    output = Path("results") / "cinematic_videos" / args.label
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = PPO.load(args.model)

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        raw_frame = env.render()
        renderer = CinematicRenderer(raw_frame.shape[1], raw_frame.shape[0], seed=args.seed + episode)
        destination = output / f"episode_{episode + 1:02d}.mp4"
        native_return, step, done = 0.0, 0, False

        with imageio.get_writer(destination, fps=args.fps, codec="libx264", quality=8) as video:
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                video.append_data(
                    renderer.render(
                        raw_frame, observation, episode, step, native_return, action=action
                    )
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                native_return += float(reward)
                step += 1
                done = terminated or truncated
                raw_frame = env.render()
            video.append_data(
                renderer.render(
                    raw_frame,
                    observation,
                    episode,
                    step,
                    native_return,
                    terminal_label(native_return, terminated),
                )
            )
        print(f"Wrote {destination}")
    env.close()


if __name__ == "__main__":
    main()
