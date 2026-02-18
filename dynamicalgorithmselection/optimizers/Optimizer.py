import time
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
from pypop7.optimizers.core import Optimizer as BaseOptimizer, Terminations


class Optimizer(BaseOptimizer):
    start_condition_parameters = ["x", "y"]

    def __init__(self, problem, options):
        BaseOptimizer.__init__(self, problem, options)
        self.best_so_far_y: float = options.get("best_so_far_y", float("inf"))
        self.best_so_far_x: Optional[np.ndarray] = None
        self._base_early_stopping: float = self.best_so_far_y
        self._counter_early_stopping: int = 0
        self.early_stopping_threshold: float = options.get(
            "early_stopping_threshold", 1e-10
        )
        self.fitness_history: List[Tuple[int, float]] = []

        self.start_conditions: Dict[str, Any] = dict()
        self.results: Dict[str, Any] = dict()

        self.worst_so_far_y: float = options.get("worst_so_far_y", -np.inf)
        self.worst_so_far_x: Optional[np.ndarray] = None
        self.x_history: List[np.ndarray] = []
        self.y_history: List[float] = []
        self.parameter_history: Dict[str, List[Any]] = {}
        self.target_FE: int | float = float("inf")

    def _evaluate_fitness(self, x, args=None, **kwargs):
        self.start_function_evaluations = time.time()
        if args is None:
            y = self.fitness_function(x)
        else:
            y = self.fitness_function(x, args=args)

        y_val = float(y)
        self.time_function_evaluations += time.time() - self.start_function_evaluations
        self.n_function_evaluations += 1

        # update best-so-far solution
        if y_val < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y_val
            self.fitness_history.append((self.n_function_evaluations, y_val))

        if y_val > self.worst_so_far_y:
            self.worst_so_far_x, self.worst_so_far_y = np.copy(x), y_val

        # update all settings related to early stopping
        if (self._base_early_stopping - y_val) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, y_val

        self.x_history.append(np.copy(x))
        self.y_history.append(y_val)

        for key, val in kwargs.items():
            if key not in self.parameter_history:
                self.parameter_history[key] = []
            if isinstance(val, np.ndarray):
                self.parameter_history[key].append(np.copy(val))
            else:
                self.parameter_history[key].append(val)

        return y_val

    def _check_success(self) -> bool:
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

    def _collect(self, fitness: List[float]) -> Dict[str, Any]:  # Added type hints
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

        for key, history in self.parameter_history.items():
            result[f"{key}_history"] = np.array(history, dtype=np.float32)
        return result

    def set_data(self, x=None, y=None, best_x=None, best_y=None, *args, **kwargs):
        n_ind = getattr(self, "n_individuals", 0)
        self.start_conditions = {
            "x": x[:n_ind] if x is not None else x,
            "y": (y[:n_ind] if isinstance(y, np.ndarray) else None),
            "best_x": best_x,
            "best_y": best_y,
        }
        self.best_so_far_x = best_x
        self.best_so_far_y = float(best_y) if best_y is not None else float("inf")

    def get_data(self, n_individuals: Optional[int] = None) -> Dict[str, Any]:
        return self.results or self.start_conditions

    def optimize(self, fitness_function=None) -> List[float]:
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness: List[float] = []
        return fitness

    def _check_terminations(self) -> bool:
        termination_signal = super()._check_terminations()
        if not termination_signal:
            termination_signal = bool(self.n_function_evaluations >= self.target_FE)
            if termination_signal:
                self.termination_signal = Terminations.MAX_FUNCTION_EVALUATIONS
        return termination_signal
