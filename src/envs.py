"""Environment factory used by training, evaluation, and video recording."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from src.rewards import HoverRewardWrapper, ProgressRewardWrapper

Condition = Literal["baseline", "hacked", "fixed"]


def make_lander(
    condition: Condition = "baseline", *, render_mode: str | None = None
) -> gym.Env:
    """Create LunarLander with the selected reward condition.

    The observation and action spaces never change. Only the reward supplied to
    PPO differs, which makes the three conditions directly comparable.
    """
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    if condition == "baseline":
        return env
    if condition == "hacked":
        return HoverRewardWrapper(env)
    if condition == "fixed":
        return ProgressRewardWrapper(env)
    raise ValueError(f"Unknown condition: {condition!r}")
