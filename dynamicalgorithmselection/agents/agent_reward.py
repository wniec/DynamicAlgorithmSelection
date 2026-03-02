from typing import Tuple

import numpy as np


class AgentReward:
    def __init__(self, option: int):
        self.reward_method = getattr(self, f"r{option}")

    def __call__(
        self,
        new_best_y: float,
        old_best_y: float,
        initial_value_range: Tuple[float, float],
        is_final_checkpoint: bool = False,
    ):
        return self.reward_method(
            new_best_y, old_best_y, initial_value_range, is_final_checkpoint
        )

    def r1(
        self,
        new_best_y: float,
        old_best_y: float,
        initial_value_range: Tuple[float, float],
        is_final_checkpoint: bool = False,
    ):
        if old_best_y == float("inf"):
            return 0.0

        improvement = old_best_y - new_best_y

        reward = improvement / (initial_value_range[1] - initial_value_range[0])
        return np.log(np.clip(reward, 0.0, 1.0) + 1e-5)

    def r2(
        self,
        new_best_y: float,
        old_best_y: float,
        initial_value_range: Tuple[float, float],
        is_final_checkpoint: bool = False,
    ):
        if old_best_y == float("inf"):
            return 0.0

        improvement = old_best_y - new_best_y

        reward = improvement / (initial_value_range[1] - initial_value_range[0])
        return np.clip(reward, 0.0, 1.0)

    def r3(
        self,
        new_best_y: float,
        old_best_y: float,
        initial_value_range: Tuple[float, float],
        is_final_checkpoint: bool = False,
    ):
        if old_best_y == float("inf") or not is_final_checkpoint:
            return 0.0

        improvement = initial_value_range[0] - new_best_y
        scale = initial_value_range[1] - initial_value_range[0]
        reward = improvement / scale
        return np.log(reward + 1e-5)

    def r4(
        self,
        new_best_y: float,
        old_best_y: float,
        initial_value_range: Tuple[float, float],
        is_final_checkpoint: bool = False,
    ):
        if old_best_y == float("inf"):
            return 0.0

        improvement = old_best_y - new_best_y

        reward = improvement / (initial_value_range[1] - initial_value_range[0])
        return 1.0 if reward >= 1e-3 else 0.0
