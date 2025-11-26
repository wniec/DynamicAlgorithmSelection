import time
from typing import Optional

import numpy as np
from pypop7.optimizers.core import Optimizer as BaseOptimizer

ALL_START_CONDITONS_PARAMETERS = ["v", "x", "y", "p_x", "p_y", "n_x", ""]


class Optimizer(BaseOptimizer):
    def __init__(self, problem, options):
        BaseOptimizer.__init__(self, problem, options)
        self.fitness_history = []
        self.start_conditions = dict()
        self.results = dict()
        self.worst_so_far_y, self.worst_so_far_x = (
            options.get("worst_so_far_y", -np.inf),
            None,
        )

    def _evaluate_fitness(self, x, args=None):
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
            self.fitness_history.append((self.n_function_evaluations, float(y)))
        if y > self.worst_so_far_y:
            self.worst_so_far_x, self.worst_so_far_y = np.copy(x), y
        # update all settings related to early stopping
        if (self._base_early_stopping - y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, y

        return float(y)

    def _check_success(self):
        if (
            (self.upper_boundary is not None)
            and (self.lower_boundary is not None)
            and self.best_so_far_x is not None
            and (
                np.any(self.lower_boundary > self.best_so_far_x)
                or np.any(self.best_so_far_x > self.upper_boundary)
            )
        ):
            return False
        elif (
            self.best_so_far_y is None
            or np.isnan(self.best_so_far_y)
            or self.best_so_far_x is None
            or np.any(np.isnan(self.best_so_far_x))
        ):
            return False
        return True

    def _collect(self, fitness):
        result = BaseOptimizer._collect(self, fitness)
        result.update(
            {
                "worst_so_far_x": self.worst_so_far_x,
                "worst_so_far_y": self.worst_so_far_y,
                "best_x": self.best_so_far_x,
                "best_y": self.best_so_far_y,
                "fitness_history": self.fitness_history,
            }
        )
        return result

    def set_data(self, x=None, y=None, best_x=None, best_y=None, *args, **kwargs):
        self.start_conditions = {
            "x": x,
            "y": (y if isinstance(y, np.ndarray) else None),
            "best_x": best_x,
            "best_y": best_y,
        }
        self.best_so_far_x = best_x
        self.best_so_far_y = best_y

    def get_data(self, n_individuals: Optional[int] = None):
        return self.results or self.start_conditions

    def optimize(self, fitness_function=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        return fitness
