import numpy as np
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer
from dynamicalgorithmselection.optimizers.PSO.PSO import PSO


class CPSO(PSO):
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)
        self.cognition = options.get("cognition", 1.49)  # cognitive learning rate
        assert self.cognition >= 0.0
        self.society = options.get("society", 1.49)  # social learning rate
        assert self.society >= 0.0
        self._max_generations = np.ceil(
            self.max_function_evaluations / (self.n_individuals * self.ndim_problem)
        )
        self._w = (
            1.0 - (np.arange(self._max_generations) + 1.0) / self._max_generations
        )  # from 1.0 to 0.0

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        fitness = []
        for j in range(self.ndim_problem):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x, fitness
            cognition_rand = self.rng_optimization.uniform(
                size=(self.n_individuals, self.ndim_problem)
            )
            society_rand = self.rng_optimization.uniform(
                size=(self.n_individuals, self.ndim_problem)
            )
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y, n_x, fitness
                n_x[i, j] = p_x[np.argmin(p_y), j]
                v[i, j] = (
                    self._w[min(self._n_generations, len(self._w) - 1)] * v[i, j]
                    + self.cognition * cognition_rand[i, j] * (p_x[i, j] - x[i, j])
                    + self.society * society_rand[i, j] * (n_x[i, j] - x[i, j])
                )  # velocity update
                v[i, j] = np.clip(v[i, j], self._min_v[j], self._max_v[j])
                x[i, j] += v[i, j]  # position update
                xx = np.copy(self.best_so_far_x)
                xx[j] = x[i, j]
                y[i] = self._evaluate_fitness(xx, args)
                fitness.append(y[i])
                if y[i] < p_y[i]:  # online update
                    p_x[i, j], p_y[i] = x[i, j], y[i]
        self._n_generations += 1
        self.results.update(
            {"v": v, "x": x, "y": y, "p_x": p_x, "p_y": p_y, "n_x": n_x}
        )
        return v, x, y, p_x, p_y, n_x, fitness

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)

        v = self.start_conditions.get("v", None)
        x = self.start_conditions.get("x", None)
        y = self.start_conditions.get("y", None)
        p_x = self.start_conditions.get("p_x", None)
        p_y = self.start_conditions.get("p_y", None)
        n_x = self.start_conditions.get("n_x", None)
        if self.best_so_far_x is None and y is not None and x is not None:
            best_so_far_idx = np.argmin(y)
            self.best_so_far_x = x[best_so_far_idx]
        v, x, y, p_x, p_y, n_x = self.initialize(args, v, x, y, p_x, p_y, n_x)
        yy = y  # only for printing
        while not self.termination_signal:
            self._print_verbose_info(fitness, yy)
            v, x, y, p_x, p_y, n_x, yy = self.iterate(v, x, y, p_x, p_y, n_x, args)
        return self._collect(fitness, yy)
