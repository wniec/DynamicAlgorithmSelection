from operator import itemgetter

import numpy as np
import torch

from dynamicalgorithmselection.agents.agent_utils import (
    get_weighted_central_moment,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class Agent(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.rewards = []
        self.choices_history = []
        self.stagnation_count = 0
        self._n_generations = 0
        self.problem = problem
        self.rewards = []
        self.options = options
        self.history = []
        self.actions = options.get("action_space")

        self.train_mode = options.get("train_mode", True)

        sub_optimization_ratio = options["sub_optimization_ratio"]
        self.run = options.get("run", None)
        self.sub_optimizer_max_fe = (
            self.max_function_evaluations / sub_optimization_ratio
        )

    def get_state(self, x: np.ndarray, y: np.ndarray) -> np.array:
        # Definition of state is inspired by its representation in DE-DDQN article
        last_action_index = (
            self.choices_history[-1] if self.choices_history else len(self.actions)
        )
        last_action_encoded = [
            float(i) for i in np.binary_repr(last_action_index, width=4)[-2:]
        ]
        same_action_counter = 0
        choices_count = [
            sum(1 for i in self.choices_history if i == j)
            / (len(self.choices_history) or 1)
            for j in range(len(self.actions))
        ]
        for i in reversed(self.choices_history):
            if i == last_action_index:
                same_action_counter += 1
            else:
                break
        if x is None or y is None:
            vector = [
                0.0,  # third weighted central moment
                0.0,  # second weighted central moment
                0.0,  # normalized domination of best solution
                0.0,  # normalized radius of the smallest sphere containing entire population
                0.5,  # normalized relative fitness difference
                0.5,  # average_y relative to best
                1.0,  # variance measure
                1.0,  # full remaining budget (max evaluations)
                self.ndim_problem / 40,  # normalized problem dimension
                0.0,  # stagnation count
                *([0.0] * 29),
                *last_action_encoded,
                *(0.0 for _ in self.actions),
                *([0.0] * 19),
            ]
        else:
            dist = lambda x1, x2: np.linalg.norm(x1 - x2)
            log_scaling = lambda x: x / (x + 5)
            dist_max = dist(self.lower_boundary, self.upper_boundary)
            x_std = 2 * x.std(axis=0) / (self.upper_boundary - self.lower_boundary)
            norm_x = (x - x.mean(axis=0)) / (x.std() + 1e-8)
            norm_dist = lambda x1, x2: min((dist(x1, x2) / dist_max), 1)
            average_y = sum(self.history) / (len(self.history) or 1)
            average_x = x.mean(axis=0)
            population_relative = x - average_x
            population_radius = np.linalg.norm(
                population_relative.max(axis=0) - population_relative.min(axis=0)
            )
            lower_x = x.min(axis=0)
            upper_x = x.max(axis=0)
            mid_so_far = (self.worst_so_far_y - self.best_so_far_y) / 2
            weights = (
                (1.0 - (y - y.min()) / (y.max() - y.min()))
                if (y.max() - y.min() > 1e-6)
                else np.ones_like(y)
            )
            weights_normalized = weights / weights.sum()
            norms = np.linalg.norm(population_relative, ord=2, axis=1)  # shape (pop,)
            second_weighted_central_moment = get_weighted_central_moment(
                2, weights_normalized, norms
            )
            third_weighted_central_moment = get_weighted_central_moment(
                3, weights_normalized, norms
            )
            i_sorted = sorted([i for i, _ in enumerate(y)], key=lambda i: y[i])
            y_normalized = (y - y.mean()) / (y.std() + 1e-6)
            y_diffs = [
                min(y_normalized[j] - y_normalized[i], 1)
                for i, j in zip(i_sorted, i_sorted[1:])
            ]
            slopes = [
                log_scaling(
                    max(y_normalized[j] - y_normalized[i], 1)
                    / (norm_dist(x[i], x[j]) + 1e-6)
                )
                for i, j in zip(i_sorted, i_sorted[1:])
            ]
            measured_individuals = list(
                itemgetter(1, 2, 3, 4, 5, 6, 9, 12, 15)(i_sorted)
            )
            choices_entropy = -(
                np.array(choices_count)
                * np.nan_to_num(np.log(choices_count), neginf=0, posinf=0, nan=0)
            ).sum() / np.log(len(choices_count))
            vector = [
                third_weighted_central_moment,
                second_weighted_central_moment,
                (y - self.best_so_far_y).mean()
                / (
                    max((y.max() - self.best_so_far_y), (y - self.best_so_far_y).sum())
                    or 1
                ),
                population_radius / dist_max,
                max(0.0, (np.min(y) - self.best_so_far_y))
                / ((self.worst_so_far_y - self.best_so_far_y) or 1),
                (average_y - self.best_so_far_y)
                / ((self.worst_so_far_y - self.best_so_far_y) or 1),
                sum(
                    (i - average_y) ** 2 for i in self.history
                )  # wcześniejsze najlepsze y
                / (self.best_so_far_y - mid_so_far)
                / (self.best_so_far_y - mid_so_far)
                / len(self.history),
                (self.max_function_evaluations - self.n_function_evaluations)
                / self.max_function_evaluations,
                self.ndim_problem
                / 40,  # maximum dimensionality in this COCO benchmark is 40
                self.stagnation_count / self.max_function_evaluations,
                *(norm_dist(x[i], self.best_so_far_x) for i in measured_individuals),
                (norm_dist(x[np.argmin(y)], self.best_so_far_x)),
                *(
                    norm_dist(x[i], average_x)
                    for i in measured_individuals + [np.argmin(y)]
                ),
                *(
                    (y[i] - np.min(y))
                    / max((self.worst_so_far_y - self.best_so_far_y), 1e-6)
                    for i in measured_individuals
                ),
                *last_action_encoded,
                same_action_counter
                / (self.max_function_evaluations / self.sub_optimizer_max_fe),
                *choices_count,
                (
                    np.prod(
                        (upper_x - lower_x)
                        / (self.upper_boundary - self.lower_boundary)
                    )
                )
                ** (1 / self.ndim_problem),  # searched volume
                x_std.max(),
                x_std.min(),
                x_std.mean(),
                2 * x_std.std(),
                np.clip((norm_x**2).mean(), -1, 1),
                np.clip((norm_x**2).min(), -1, 1),
                np.clip((norm_x**2).max(), -1, 1),
                np.clip((norm_x**2).std(), -1, 1),
                choices_entropy,
                norm_dist(self.best_so_far_x, self.worst_so_far_x),
                norm_dist(x[np.argmin(y)], x[np.argmax(y)]),
                max(y_diffs),
                min(y_diffs),
                sum(y_diffs) / len(y_diffs),
                max(slopes),
                min(slopes),
                sum(slopes) / len(slopes),
            ]
        return torch.tensor(vector, dtype=torch.float)

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and (
            (not self._n_generations % self.verbose) or (self.termination_signal > 0)
        ):
            info = "  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}"
            print(
                info.format(
                    self._n_generations,
                    self.best_so_far_y,
                    np.min(y),
                    self.n_function_evaluations,
                )
            )

    def _save_fitness(self, best_x, best_y, worst_x, worst_y):
        # update best-so-far solution (x) and fitness (y)
        self.history.append(best_y)
        if best_y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(best_x), best_y
        if worst_y > self.worst_so_far_y:
            self.worst_so_far_x, self.worst_so_far_y = np.copy(worst_x), worst_y
        # update all settings related to early stopping
        if (self._base_early_stopping - best_y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, best_y

    def iterate(self, optimizer_input_data=None, optimizer=None):
        optimizer_input_data["best_x"] = self.best_so_far_x
        optimizer_input_data["best_y"] = self.best_so_far_y
        optimizer.set_data(**optimizer_input_data)
        if self._check_terminations():
            return optimizer.get_data()
        self._n_generations += 1
        results = optimizer.optimize()
        self._save_fitness(
            results["best_so_far_x"],
            results["best_so_far_y"],
            results["worst_so_far_x"],
            results["worst_so_far_y"],
        )  # fitness evaluation
        return optimizer.get_data()

    def _collect(self, fitness, y=None):
        raise NotImplementedError

    def optimize(self, fitness_function=None, args=None):
        raise NotImplementedError

    def get_reward(self, y, best_parent):
        log_scale = lambda x: np.log(np.clip(x, a_min=0, a_max=None) + 1)
        reference = max(
            self.worst_so_far_y
            - (self.best_so_far_y if best_parent == float("inf") else best_parent),
            1e-5,
        )
        best_individual = np.min(y)
        improvement = (
            (best_parent - best_individual) if best_individual is not None else 0
        )
        # used_fe = self.n_function_evaluations / self.max_function_evaluations
        reward = log_scale(improvement) / log_scale(reference)
        if len(self.choices_history) > 1:
            pass
            # reward += 0.05 if self.choices_history[-1] == self.choices_history[-2] else 0.0
        else:
            return 0
        # reward = np.sign(improvement)#  * used_fe
        return np.clip(np.cbrt(reward), a_min=-0.0, a_max=0.5)  # to the 1/dim power ?
