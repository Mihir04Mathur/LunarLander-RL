"""Reward wrappers for the reward-hacking experiment.

Each wrapper records the untouched environment reward in ``info['native_reward']``.
This makes it possible to plot learned reward separately from task performance.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np


class HoverRewardWrapper(gym.Wrapper):
    """Intentionally exploitable reward: pays for a good state every timestep."""

    def __init__(self, env: gym.Env, bonus: float = 1.0) -> None:
        super().__init__(env)
        self.bonus = bonus

    def step(self, action: Any):
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = obs
        upright_and_slow = abs(angle) < 0.10 and abs(vx) < 0.10 and abs(vy) < 0.10
        shaping_bonus = self.bonus if upright_and_slow else 0.0
        shaped_reward = native_reward + shaping_bonus
        info = dict(info)
        info.update(
            native_reward=float(native_reward),
            shaping_bonus=float(shaping_bonus),
            time_cost=0.0,
        )
        return obs, shaped_reward, terminated, truncated, info


class ProgressRewardWrapper(gym.Wrapper):
    """Reward progress toward a controlled landing and discourage waiting.

    It uses a potential difference rather than a per-step state bonus, so a
    lander that stays still cannot keep collecting a large positive reward.
    """

    def __init__(
        self, env: gym.Env, shaping_scale: float = 2.0, time_cost: float = 0.01
    ) -> None:
        super().__init__(env)
        self.shaping_scale = shaping_scale
        self.time_cost = time_cost
        self.previous_potential: float | None = None

    @staticmethod
    def potential(observation: np.ndarray) -> float:
        """Return a higher value for a closer, slower, more upright lander."""
        x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
        distance = math.hypot(float(x), float(y))
        speed = math.hypot(float(vx), float(vy))
        return -(distance + 0.7 * speed + 0.3 * abs(float(angle)))

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(observation)
        return observation, info

    def step(self, action: Any):
        observation, native_reward, terminated, truncated, info = self.env.step(action)
        current_potential = self.potential(observation)
        if self.previous_potential is None:
            self.previous_potential = current_potential

        progress_bonus = self.shaping_scale * (
            current_potential - self.previous_potential
        )
        self.previous_potential = current_potential
        shaped_reward = native_reward + progress_bonus - self.time_cost

        info = dict(info)
        info.update(
            native_reward=float(native_reward),
            shaping_bonus=float(progress_bonus),
            time_cost=float(self.time_cost),
        )
        return observation, shaped_reward, terminated, truncated, info
