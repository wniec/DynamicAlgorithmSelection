import numpy as np

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class IDENTITY(Optimizer):
    """An optimizer that returns the same x, y it received — a no-op pass-through."""

    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.n_individuals = 1

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x = self.start_conditions.get("x", None)

        if x is None:
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                size=(1, self.ndim_problem),
            )

        while not self._check_terminations():
            self._evaluate_fitness(x[0])

        return self._collect(fitness)

    def _collect(self, fitness):
        for i in range(1, len(fitness)):
            if np.isnan(fitness[i]):
                fitness[i] = fitness[i - 1]
        results = Optimizer._collect(self, fitness)
        return results
