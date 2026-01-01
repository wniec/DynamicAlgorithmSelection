from operator import itemgetter
import numpy as np
import torch

from dynamicalgorithmselection.agents.agent_utils import (
    distance,
    inverse_scaling,
    get_list_stats, MAX_POP_DIM,
)


class AgentState:
    def __init__(
        self,
        x,
        y,
        best_x,
        best_y,
        lower_bound,
        upper_bound,
        worst_x,
        worst_y,
        y_history,
        choice_history,
        n_actions,
    ):
        self.x = x
        self.y = y
        self.best_x = best_x
        self.best_y = best_y
        self.worst_x = worst_x
        self.worst_y = worst_y
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.y_history = y_history
        self.choice_history = choice_history
        self.n_actions = n_actions

        self.y_normalized = (y - y.mean()) / (y.std() + 1e-6)
        self.max_distance = distance(self.lower_bound, self.upper_bound)
        self.sorted_indices = sorted(
            [i for i, _ in enumerate(y)], key=lambda i: y[i]
        )  # population indices sorted by fitness
        self.measured_individuals = list(
            itemgetter(*(min(i, len(y) - 1) for i in (1, 2, 3, 4, 5, 6, 9, 12, 15)))(
                self.sorted_indices
            )
        )
        self.x_mean = x.mean(axis=0)
        self.population_relative = x - self.x_mean
        self.normalized_x = (x - x.mean(axis=0)) / (x.std() + 1e-8)
        self.x_std = 2 * x.std(axis=0) / (self.upper_bound - self.lower_bound)

        self.mean_historic_y = sum(self.y_history) / (len(self.y_history) or 1)
        self.last_action_index = (
            self.choice_history[-1] if self.choice_history else None
        )
        self.last_action_encoded = [0 for _ in range(n_actions)]

        if self.last_action_index is not None:
            self.last_action_encoded[self.last_action_index] = 1

        self.choices_frequency = [
            sum(1 for i in self.choice_history if i == j)
            / (len(self.choice_history) or 1)
            for j in range(self.n_actions)
        ]

    def get_weighted_central_moment(self, n: int):
        norms_squared = np.linalg.norm(
            self.population_relative, ord=2, axis=1
        )  # shape (pop,)
        weights = self.get_fitness_weights()
        exponent = n / 2
        numerator = min((weights * norms_squared**exponent).sum(), 1e8)
        inertia_denom_w = np.linalg.norm(weights)
        inertia_denom_n = np.linalg.norm(norms_squared**exponent)
        return numerator / max(1e-5, inertia_denom_w * max(1e-5, inertia_denom_n))

    def normalized_distance(self, x0: np.ndarray, x1: np.ndarray) -> float:
        return min(np.linalg.norm(x0 - x1) / self.max_distance, 1.0)

    def get_fitness_weights(self) -> np.ndarray:
        weights = (
            (1.0 - (self.y - self.y.min()) / (self.y.max() - self.y.min()))
            if (self.y.max() - self.y.min() > 1e-6)
            else np.ones_like(self.y)
        )
        return weights / weights.sum()

    def population_relative_radius(self) -> np.ndarray:
        population_radius = np.linalg.norm(self.x.max(axis=0) - self.x.min(axis=0))
        return population_radius / self.max_distance

    def slopes_stats(self) -> tuple:
        return get_list_stats(
            [
                inverse_scaling(
                    max(self.y_normalized[j] - self.y_normalized[i], 1)
                    / (self.normalized_distance(self.x[i], self.x[j]) + 1e-6)
                )
                for i, j in zip(self.sorted_indices, self.sorted_indices[1:])
            ]
        )

    def y_difference_stats(self) -> tuple:
        return get_list_stats(
            [
                min(self.y_normalized[j] - self.y_normalized[i], 1)
                for i, j in zip(self.sorted_indices, self.sorted_indices[1:])
            ]
        )

    def distances_from_best(self) -> list:
        return [
            self.normalized_distance(self.x[i], self.best_x)
            for i in self.measured_individuals + [0, -1]
        ]

    def distances_from_mean(self) -> list:
        return [
            self.normalized_distance(self.x[i], self.x_mean)
            for i in self.measured_individuals + [0, -1]
        ]

    def explored_volume(self) -> float:
        return np.prod(
            (self.x.max(axis=0) - self.x.min(axis=0))
            / (self.upper_bound - self.lower_bound)
        )

    def relative_improvement(self):
        return max(0.0, (np.min(self.y) - self.best_y)) / (
            (self.worst_y - self.best_y) or 1.0
        )

    def normalized_x_stats(self) -> tuple:
        return (
            np.clip((self.normalized_x**2).mean(), -1, 1),
            np.clip((self.normalized_x**2).min(), -1, 1),
            np.clip((self.normalized_x**2).max(), -1, 1),
            np.clip((self.normalized_x**2).std(), -1, 1),
        )

    def relative_y_differences(self) -> list:
        return [
            (self.y[i] - np.min(self.y)) / max((self.worst_y - self.best_y), 1e-6)
            for i in self.measured_individuals
        ]

    def x_standard_deviation_stats(self) -> tuple:
        return (
            self.x_std.max(),
            self.x_std.min(),
            self.x_std.mean(),
            2 * self.x_std.std(),
        )

    def y_historic_improvement(self) -> float:
        return (self.mean_historic_y - self.best_y) / (
            (self.worst_y - self.best_y) or 1
        )

    def y_deviation(self) -> float:
        middle_y = (self.worst_y - self.best_y) / 2
        max_possible_std = self.best_y - middle_y
        # dividing twice by std instead of variance due to numerical instability issues
        return (
            sum((i - self.mean_historic_y) ** 2 for i in self.y_history)
            / len(self.y_history)
            / max_possible_std
            / max_possible_std
        )

    def choice_entropy(self) -> float:
        return -(
            np.array(self.choices_frequency)
            * np.nan_to_num(np.log(self.choices_frequency), neginf=0, posinf=0, nan=0)
        ).sum() / np.log(len(self.choices_frequency))

    def same_action_counter(self) -> int:
        same_action_counter = 0
        for i in reversed(self.choice_history):
            if i == self.last_action_index:
                same_action_counter += 1
            else:
                break
        return same_action_counter

    def mean_falling_behind(self) -> float:
        return max(
            (self.y - self.best_y).mean() / (max((self.y.max() - self.best_y), 1e-8)), 0
        )


def get_padded_population_observation(x, y, pop_size):

    padded_obs = np.zeros((pop_size, 1 + MAX_POP_DIM), dtype=np.float32)
    if x is None or y is None:
        return torch.tensor(padded_obs, dtype=torch.float)
    best_indices = sorted(
        list(range(len(y))), key=lambda idx: y[idx]
    )[: pop_size]
    considered_x = x[best_indices]
    considered_y = y[best_indices]
    real_dim = x.shape[1]
    padded_obs[:, 0] = considered_y
    padded_obs[:, 1:1 + real_dim] = considered_x
    return torch.tensor(padded_obs, dtype=torch.float)
