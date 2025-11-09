import numpy as np
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class DS(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.x = options.get(
            "x", np.random.uniform(low=self.lower_boundary, high=self.upper_boundary)
        )
        # initial (starting) point
        self.sigma = options.get("sigma", 1.0)  # initial global step-size
        assert self.sigma > 0.0
        self._n_generations = 0  # number of generations
        # set for restart
        self.sigma_threshold = options.get(
            "sigma_threshold", 1e-12
        )  # stopping threshold of sigma for restart
        assert self.sigma_threshold >= 0.0
        self.stagnation = options.get("stagnation", np.maximum(30, self.ndim_problem))
        assert self.stagnation > 0
        self.fitness_diff = options.get(
            "fitness_diff", 1e-12
        )  # stopping threshold of fitness difference for restart
        assert self.fitness_diff >= 0.0
        self._sigma_bak = np.copy(self.sigma)  # bak for restart
        self._fitness_list = [
            self.best_so_far_y
        ]  # to store `best_so_far_y` generated in each generation
        self._n_restart = 0  # number of restarts
        self._printed_evaluations = self.n_function_evaluations

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_x(self, is_restart=False):
        if is_restart or (self.x is None):
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary
            )
        else:
            x = np.copy(self.x)
        assert x.shape == (self.ndim_problem,)
        return x

    def _print_verbose_info(self, fitness, y, is_print=False):
        if y is not None:
            if self.saving_fitness:
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

    def _collect(self, fitness, y=None):
        if len(y) > 0:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results
