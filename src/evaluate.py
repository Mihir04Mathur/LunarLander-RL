"""Evaluate a model strictly on the original, unwrapped LunarLander reward."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.envs import make_lander


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path without or with the .zip suffix.")
    parser.add_argument("--label", required=True, help="Experiment label for output files.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # No reward wrapper here: this is the source-of-truth task evaluation.
    env = make_lander("baseline")
    model = PPO.load(args.model)
    rows: list[dict[str, float | int]] = []

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rows.append({"episode": episode, "native_return": total_reward, "steps": steps})

    env.close()
    output_dir = Path("results") / "native_evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "native_return", "steps"])
        writer.writeheader()
        writer.writerows(rows)

    returns = np.array([float(row["native_return"]) for row in rows])
    steps = np.array([int(row["steps"]) for row in rows])
    summary = {
        "label": args.label,
        "episodes": args.episodes,
        "mean_native_return": float(returns.mean()),
        "std_native_return": float(returns.std()),
        "success_rate_return_ge_200": float((returns >= 200).mean()),
        "mean_episode_steps": float(steps.mean()),
    }
    summary_path = output_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
