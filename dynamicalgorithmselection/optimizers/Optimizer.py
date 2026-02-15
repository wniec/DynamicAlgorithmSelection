import time
from typing import Optional

import numpy as np
from pypop7.optimizers.core import Optimizer as BaseOptimizer, Terminations  # type: ignore


class Optimizer(BaseOptimizer):
    start_condition_parameters = ["x", "y"]

    def __init__(self, problem, options):
        BaseOptimizer.__init__(self, problem, options)
        self.fitness_history = []
        self.start_conditions = dict()
        self.results = dict()
        self.worst_so_far_y, self.worst_so_far_x = (
            options.get("worst_so_far_y", -np.inf),
            None,
        )
        self.x_history, self.y_history = [], []
        # [Added] Dictionary to store histories of generic parameters
        self.parameter_history = {}
        self.target_FE: int | float = float("inf")

    # [Modified] Accept generic kwargs for history tracking
    def _evaluate_fitness(self, x, args=None, **kwargs):
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

        self.x_history.append(np.copy(x))
        self.y_history.append(float(y))

        # [Added] Generic storage for any extra parameters passed
        for key, val in kwargs.items():
            if key not in self.parameter_history:
                self.parameter_history[key] = []

            # Store copy if it's an array to prevent reference issues
            if isinstance(val, np.ndarray):
                self.parameter_history[key].append(np.copy(val))
            else:
                self.parameter_history[key].append(val)

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
                "x_history": np.array(self.x_history, dtype=np.float32),
                "y_history": np.array(self.y_history, dtype=np.float32),
            }
        )

        # [Added] Inject generic parameter histories into result
        # Keys will be named like 'v_history', 'p_x_history' automatically
        for key, history in self.parameter_history.items():
            result[f"{key}_history"] = np.array(history, dtype=np.float32)
        return result

    def set_data(self, x=None, y=None, best_x=None, best_y=None, *args, **kwargs):
        self.start_conditions = {
            "x": x[: self.n_individuals] if x is not None else x,
            "y": (y[: self.n_individuals] if isinstance(y, np.ndarray) else None),
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

    def _check_terminations(self):
        termination_signal = super()._check_terminations()
        if not termination_signal:
            termination_signal = self.n_function_evaluations >= self.target_FE
            self.termination_signal = Terminations.MAX_FUNCTION_EVALUATIONS
        return termination_signal
