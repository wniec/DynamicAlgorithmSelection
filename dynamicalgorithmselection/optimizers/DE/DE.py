import numpy as np  # engine for numerical computing
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class DE(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if (
            self.n_individuals is None
        ):  # number of offspring, aka offspring population size
            self.n_individuals = 170
        assert self.n_individuals > 0
        self._n_generations = 0  # number of generations
        self._printed_evaluations = self.n_function_evaluations

    def initialize(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _print_verbose_info(self, fitness, y, is_print=False):
        if y is not None and self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose:
            is_verbose = (
                self._printed_evaluations != self.n_function_evaluations
            )  # to avoid repeated printing
            is_verbose_1 = (not self._n_generations % self.verbose) and is_verbose
            is_verbose_2 = self.termination_signal > 0 and is_verbose
            is_verbose_3 = is_print and is_verbose
            if is_verbose_1 or is_verbose_2 or is_verbose_3:
                info = "  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}"
                print(
                    info.format(
                        self._n_generations,
                        self.best_so_far_y,
                        np.min(y),
                        self.n_function_evaluations,
                    )
                )
                self._printed_evaluations = self.n_function_evaluations

    def _collect(self, fitness=None, y=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        results.update(self.results)
        return results
