import numpy as np

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class GA(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size
            self.n_individuals = 100
        assert self.n_individuals > 0
        self._n_generations = 0

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

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

    def _collect(self, fitness, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results
