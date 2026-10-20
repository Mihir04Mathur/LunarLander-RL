"""Train PPO for a selected reward condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from src.envs import make_lander


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "hacked", "fixed"], required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = f"{args.condition}_seed{args.seed}"
    model_dir = Path("models") / run_name
    result_dir = Path("results") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    factory = lambda: make_lander(args.condition)
    train_env = make_vec_env(factory, n_envs=args.n_envs, seed=args.seed, monitor_dir=str(result_dir / "monitor"))
    eval_env = make_vec_env(factory, n_envs=1, seed=args.seed + 10_000)

    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(result_dir / "evaluations"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )
    checkpoints = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo",
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(result_dir / "tensorboard"),
    )
    model.learn(total_timesteps=args.timesteps, callback=[evaluation, checkpoints])
    model.save(str(model_dir / "final_model"))

    (result_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
