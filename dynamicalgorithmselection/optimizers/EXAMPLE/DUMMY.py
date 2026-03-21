import numpy as np

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class DUMMY(Optimizer):
    """An optimizer, that always selects the same x for evaluation."""

    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.initialized = False
        self.x, self.y = None, None
        self.n_individuals = 1

    def initialize(self, args=None, x=None, y=None):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary,
            self.initial_upper_boundary,
        )  # population
        y = self._evaluate_fitness(x, args)
        self.initialized = True
        self.x, self.y = x, y
        return x, y

    def _collect(self, fitness, y=None):
        for i in range(1, len(fitness)):  # to avoid `np.nan`
            if np.isnan(fitness[i]):
                fitness[i] = fitness[i - 1]
        results = Optimizer._collect(self, fitness)
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y = (
            self.start_conditions.get("x", None),
            self.start_conditions.get("y", None),
        )
        x, y = self.initialize(args, x, y)
        while not self._check_terminations():
            self.y = self._evaluate_fitness(self.x)
            self._evaluate_fitness(x, args)
        return self._collect(fitness, y)
