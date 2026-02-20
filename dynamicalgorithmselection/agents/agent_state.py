import warnings
from operator import itemgetter
from typing import Tuple, Callable, Any

import numpy as np
import pandas as pd
from pflacco.classical_ela_features import (
    calculate_ela_meta,  # Meta-Model (Linear/Quadratic fit)
    calculate_nbc,  # Nearest Better Clustering
    calculate_dispersion,  # Dispersion of good solutions
    calculate_information_content,
    calculate_ela_distribution,  # Information Content
)
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from dynamicalgorithmselection.NeurELA.NeurELA import feature_embedder
from dynamicalgorithmselection.agents.agent_utils import MAX_DIM, RunningMeanStd

BASE_STATE_SIZE = 102
MAX_CONSIDERED_POPSIZE = 2500


def get_state_representation(
    name: str, n_actions: int
) -> Tuple[Callable[[np.ndarray, np.ndarray, Any], np.ndarray], int]:
    """
    :param name: name of the state representation mode
    :param n_actions: number of actions to take
    :return: function used to infer state representation from population and dimensionality of that state representation
    """
    if name == "NeurELA":
        return lambda x, y, *args: feature_embedder(
            x[-MAX_CONSIDERED_POPSIZE:], y[-MAX_CONSIDERED_POPSIZE:]
        )[0].mean(axis=0), 34
    elif name == "ELA":
        return lambda x, y, *args: ela_state_representation(x, y), 47
    elif name == "custom":
        return lambda x, y, args: AgentState(
            x, y, n_actions, *args
        ).get_state(), BASE_STATE_SIZE + 2 * n_actions + 2
    else:
        raise ValueError("incorrect state representation")


def ela_state_representation(x, y, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        _, unique_indices = np.unique(x, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        x_deduplicated = x[unique_indices][-MAX_CONSIDERED_POPSIZE:]
        y_deduplicated = y[unique_indices][-MAX_CONSIDERED_POPSIZE:]

        x_raw = np.ascontiguousarray(x_deduplicated - x_deduplicated.mean()) / (
            x_deduplicated.std() + 1e-8
        )
        y_raw = np.ascontiguousarray(y_deduplicated - y_deduplicated.mean()) / (
            y_deduplicated.std() + 1e-8
        )

        x_norm = pd.DataFrame(x_raw).reset_index(drop=True)
        x_norm.columns = [f"x_{i}" for i in range(x_norm.shape[1])]
        y_norm = pd.Series(y_raw).reset_index(drop=True)

        is_unique = ~x_norm.duplicated()

        # If we lost data, re-slice to ensure alignment
        if not is_unique.all():
            x_norm = x_norm[is_unique].reset_index(drop=True)
            y_norm = y_norm[is_unique].reset_index(drop=True)

        meta_feats = calculate_ela_meta(x_norm, y_norm)
        ela_distr = (
            calculate_ela_distribution(x_norm, y_norm)
            if ((y**2).sum() > 0 and np.var(y_norm) > 1e-8)
            else {str(i): 0 for i in range(4)}
        )
        nbc_feats = calculate_nbc(x_norm, y_norm)
        disp_feats = calculate_dispersion(x_norm, y_norm)

        ic_feats = calculate_information_content(x_norm, y_norm)

        all_features = {
            **meta_feats,
            **nbc_feats,
            **disp_feats,
            **ic_feats,
            **ela_distr,
        }
        return np.array(list(all_features.values()), dtype=np.float32)


class AgentState:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_actions,
        lower_bound,
        upper_bound,
        choice_history,
        n_checkpoints,
        n_dim_problem,
    ):
        self.x = x
        self.y = y
        self.n_actions = n_actions
        self.n_checkpoints = n_checkpoints
        self.ndim_problem = n_dim_problem

        if x is None:
            return

        best_idx = y.argmin()
        worst_idx = y.argmax()

        self.best_x: np.ndarray = x[best_idx]
        self.best_y: float = y[best_idx]
        self.worst_x: np.ndarray = x[worst_idx]
        self.worst_y: float = y[worst_idx]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.choice_history = choice_history

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

        self.mean_historic_y = y.mean()
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
        return numerator / max(1e-5, inertia_denom_w * inertia_denom_n)

    def normalized_distance(self, x0: np.ndarray, x1: np.ndarray) -> float:
        return float(min(np.linalg.norm(x0 - x1) / self.max_distance, 1.0))

    def get_fitness_weights(self) -> np.ndarray:
        weights = (
            (1.0 - (self.y - self.y.min()) / (self.y.max() - self.y.min()))
            if (self.y.max() - self.y.min() > 1e-6)
            else np.ones_like(self.y)
        )
        return weights / weights.sum()

    def population_relative_radius(self) -> float:
        population_radius = np.linalg.norm(self.x.max(axis=0) - self.x.min(axis=0))
        return float(population_radius / self.max_distance)

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
            sum((i - self.mean_historic_y) ** 2 for i in self.y)
            / len(self.y)
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
        return (self.y - self.best_y).mean() / (
            max((self.y.max() - self.best_y), (self.y - self.best_y).mean()) or 1
        )

    def get_initial_state(self):
        vector = [
            0.0,  # third weighted central moment
            0.0,  # second weighted central moment
            0.0,  # normalized domination of best solution
            0.0,  # normalized radius of the smallest sphere containing entire population
            0.5,  # normalized relative fitness difference
            0.5,  # average_y relative to best
            1.0,  # normalized y deviation measure
            1.0,  # full remaining budget (max evaluations)
            0.0,  # stagnation count
            *([0.0] * (51 + 2 * self.n_actions)),
            self.ndim_problem / 40,  # normalized problem dimension
        ]
        return np.array(vector, dtype=np.float32)

    def get_state(self, optimization_status=False) -> np.ndarray:
        if len(self.x) < 1:
            return self.get_initial_state()
        else:
            vector = [
                self.get_weighted_central_moment(3),
                self.get_weighted_central_moment(2),
                self.mean_falling_behind(),
                self.population_relative_radius(),
                self.relative_improvement(),
                self.y_historic_improvement(),
                self.y_deviation(),
                *self.distances_from_best(),
                *self.distances_from_mean(),
                *self.relative_y_differences(),
                *(self.last_action_encoded if optimization_status else ()),
                *(
                    (self.same_action_counter() / self.n_checkpoints,)
                    if optimization_status
                    else ()
                ),
                *(self.choices_frequency if optimization_status else ()),
                self.explored_volume() ** (1 / self.ndim_problem),  # searched volume
                *self.x_standard_deviation_stats(),
                *self.normalized_x_stats(),
                *((self.choice_entropy(),) if optimization_status else ()),
                self.normalized_distance(self.best_x, self.worst_x),
                *self.y_difference_stats(),
                *self.slopes_stats(),
                *((self.ndim_problem / MAX_DIM,) if optimization_status else ()),
            ]
        return np.array(vector, dtype=np.float32)


def distance(x0: np.ndarray, x1: np.ndarray) -> float:
    return float(np.linalg.norm(x0 - x1))


def inverse_scaling(x):
    # Monotonic increacing in [0, inf) function that is bounded in [0, 1)
    return x / (x + 5)


def get_list_stats(data: list):
    return (
        max(data),
        min(data),
        sum(data) / len(data),
    )


class StateNormalizer:
    def __init__(self, input_shape):
        self.rms = RunningMeanStd(shape=input_shape)

    def normalize(self, state, update=True):
        """
        Normalizes the state: (state - mean) / std.

        Args:
            state (np.array): The input state vector.
            update (bool): Whether to update the running statistics.
                           Usually True during training, False during testing.
        """
        state = np.asarray(state)

        if update:
            if len(state.shape) == 1:
                self.rms.update(state.reshape(1, -1))
            else:
                self.rms.update(state)

        std = np.sqrt(self.rms.var) + 1e-8

        normalized_state = (state - self.rms.mean) / std
        return np.clip(normalized_state, -5.0, 5.0)


def get_la_features(agent, pop_x, pop_y):
    """
    Extracts 9 Landscape Analysis features described in Reinforcement Learning Dynamic Algorithm Selection.
    Includes sampling-based features (f5-f8) which consume function evaluations.
    """
    sorted_idx = np.argsort(pop_y)
    pop_x = pop_x[sorted_idx]
    pop_y = pop_y[sorted_idx]

    best_y = pop_y[0]
    best_x = pop_x[0]
    n = len(pop_x)

    norm_factor = (
        agent.initial_cost
        if agent.initial_cost and abs(agent.initial_cost) > 1e-9
        else 1.0
    )
    f1 = best_y / norm_factor

    dists_to_best = np.linalg.norm(pop_x - best_x, axis=1)
    if np.std(pop_y) < 1e-9 or np.std(dists_to_best) < 1e-9:
        f2 = 0.0
    else:
        fdc, _ = spearmanr(pop_y, dists_to_best)
        f2 = fdc if not np.isnan(fdc) else 0.0

    n_top = max(2, int(0.1 * n))

    if n > 1:
        dist_matrix_all = pdist(pop_x)
        disp_all = np.mean(dist_matrix_all) if len(dist_matrix_all) > 0 else 0.0

        dist_matrix_top = pdist(pop_x[:n_top])
        disp_top = np.mean(dist_matrix_top) if len(dist_matrix_top) > 0 else 0.0

        f3 = disp_all - disp_top
        f4 = np.max(dist_matrix_all) if len(dist_matrix_all) > 0 else 0.0
    else:
        f3, f4 = 0.0, 0.0

    remaining_fes = agent.max_function_evaluations - agent.n_function_evaluations
    cost_per_sample = n  # 1 generation of size N

    sampled_pops_y = []

    if remaining_fes >= (2 * cost_per_sample):
        sample_indices = np.random.choice(len(agent.actions), 2, replace=False)

        for idx in sample_indices:
            alg_class = agent.actions[idx]

            sub_opt = alg_class(agent.problem, agent.options)

            sub_opt.population = pop_x.copy()
            sub_opt.fitness = pop_y.copy()

            sub_opt.n_function_evaluations = 0
            sub_opt.max_function_evaluations = cost_per_sample

            sub_opt.optimize()

            sampled_pops_y.append(sub_opt.fitness)
            agent.n_function_evaluations += sub_opt.n_function_evaluations

    f5, f6, f7, f8 = 0.0, 0.0, 0.0, 0.0

    if len(sampled_pops_y) > 0:
        sorted_current = np.sort(pop_y)
        sorted_samples = [np.sort(sy) for sy in sampled_pops_y]
        avg_sample_y = np.mean(sorted_samples, axis=0)

        # Slopes: (y_{i+1} - y_i)
        diff_current = np.diff(sorted_current)
        diff_sample = np.diff(avg_sample_y)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = diff_current / diff_sample
            ratios[diff_sample == 0] = 0.0
            ratios[np.isnan(ratios)] = 0.0

        f5 = min(np.sum(ratios), 0.0)

        S = len(sampled_pops_y)
        eps = 1e-8

        neutral_count = 0
        no_improve_counts = np.zeros(n)  # For f7
        all_worse_counts = np.zeros(n)  # For f8

        for sy in sampled_pops_y:
            neutral_count += np.sum(np.abs(pop_y - sy) < eps)

            improved = sy < pop_y
            no_improve_counts += improved.astype(int)  # Add 1 if improved

            worse = sy > pop_y
            all_worse_counts += worse.astype(int)

        f6 = neutral_count / (n * S)

        alphas = (no_improve_counts == 0).astype(float)
        f7 = np.mean(alphas)

        betas = (all_worse_counts == S).astype(float)
        f8 = np.mean(betas)

    f9 = agent.n_function_evaluations / agent.max_function_evaluations

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])
