from itertools import product
from typing import List, Type, Optional
import numpy as np
from dynamicalgorithmselection.agents.agent_state import (
    get_state_representation,
    StateNormalizer,
)
from dynamicalgorithmselection.agents.agent_utils import (
    get_checkpoints,
    StepwiseRewardNormalizer,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer
from dynamicalgorithmselection.optimizers.RestartOptimizer import restart_optimizer


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
        self.actions: List[Type[Optimizer]] = options.get("action_space") + (
            [restart_optimizer(i) for i in options.get("action_space")]
            if options.get("force_restarts")
            else []
        )
        self.name = options.get("name")
        self.cdb = options.get("cdb")

        self.train_mode = options.get("train_mode", True)

        self.n_checkpoints = options["n_checkpoints"]
        self.run = options.get("run", None)
        self.checkpoints = get_checkpoints(
            self.n_checkpoints,
            self.max_function_evaluations,
            self.n_individuals,
            self.cdb,
        )
        self.reward_normalizer = self.options.get(
            "reward_normalizer", StepwiseRewardNormalizer(max_steps=self.n_checkpoints)
        )
        self.state_representation, self.state_dim = get_state_representation(
            self.options.get("state_representation", None), len(self.actions)
        )
        self.state_normalizer = StateNormalizer(input_shape=(self.state_dim,))

    def get_state(
        self, x: Optional[np.ndarray], y: Optional[np.ndarray], train_mode: bool
    ) -> np.array:
        if x is None or y is None:
            state_representation = self.state_representation(
                np.zeros((50, self.ndim_problem)),
                np.zeros((50,)),
                (
                    self.lower_boundary,
                    self.upper_boundary,
                    self.choices_history,
                    self.n_checkpoints,
                    self.ndim_problem,
                ),
            )
            state_representation = np.append(state_representation, (0, 0))
            return self.state_normalizer.normalize(
                state_representation, update=train_mode
            )
        used_fe = self.n_function_evaluations / self.max_function_evaluations
        # best_idx = sorted(range(len(y)), key=lambda i: y[i])[: self.n_individuals]
        stagnation_coef = self.stagnation_count / self.max_function_evaluations
        sr_additional_params = (
            self.lower_boundary,
            self.upper_boundary,
            self.choices_history,
            self.n_checkpoints,
            self.ndim_problem,
        )
        state_representation = self.state_representation(x, y, sr_additional_params)
        state_representation = np.append(
            state_representation, (used_fe, stagnation_coef)
        )
        return self.state_normalizer.normalize(state_representation, update=train_mode)

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
        self.best_parent = best_y
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
        self.fitness_history.extend(results["fitness_history"])
        self._save_fitness(
            results["best_so_far_x"],
            results["best_so_far_y"],
            results["worst_so_far_x"],
            results["worst_so_far_y"],
        )  # fitness evaluation
        return optimizer.get_data(self.n_individuals) | {
            "x_history": results["x_history"],
            "y_history": results["y_history"],
        }

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        results["fitness_history"] = self.fitness_history
        if self.run:
            choices_count = {
                self.actions[j].__name__: sum(1 for i in self.choices_history if i == j)
                / (len(self.choices_history) or 1)
                for j in range(len(self.actions))
            }
            self.run.log(choices_count)
            self.run.log(
                {f"{k}_dim_{self.ndim_problem}": v for k, v in choices_count.items()},
            )
            checkpoint_choices = {
                f"{action.__name__}_checkpoint{i}": (
                    1 if self.choices_history[i] == action_id else 0
                )
                for i, (action_id, action) in product(
                    range(self.n_checkpoints), enumerate(self.actions)
                )
            }
            self.run.log(checkpoint_choices)
        return results, None

    def optimize(self, fitness_function=None, args=None):
        raise NotImplementedError

    def get_reward(self, new_best_y, old_best_y):
        value_range = max(
            self.worst_so_far_y
            - (self.best_so_far_y if old_best_y == float("inf") else old_best_y),
            1e-5,
        )
        improvement = old_best_y - new_best_y

        if len(self.choices_history) > 1:
            reward = improvement / value_range
        else:
            return 0.0
        reward = min(reward, 1.0)
        return reward
