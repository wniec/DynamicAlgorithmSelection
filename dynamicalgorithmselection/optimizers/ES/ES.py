import numpy as np

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class ES(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        # population size, sample size, number of offspring (Nikolaus Hansen, 2023)
        if (
            self.n_individuals is None
        ):  # number of offspring (λ: lambda), offspring population size
            self.n_individuals = 4 + int(
                3 * np.log(self.ndim_problem)
            )  # only for small populations setting
        assert (
            self.n_individuals > 0
        ), f"`self.n_individuals` = {self.n_individuals}, but should > 0."
        # parent number, number of (positively) selected search points in the population,
        #   number of strictly positive recombination weights (Nikolaus Hansen, 2023)
        if (
            self.n_parents is None
        ):  # number of parents (μ: mu), parental population size
            self.n_parents = int(self.n_individuals / 2)
        assert (
            self.n_parents <= self.n_individuals
        ), f"self.n_parents (== {self.n_parents}) should <= self.n_individuals (== {self.n_individuals})"
        if self.n_parents > 1:
            self._w, self._mu_eff = self._compute_weights()
            self._e_chi = np.sqrt(
                self.ndim_problem
            ) * (  # E[||N(0,I)||]: expectation of chi distribution
                1.0
                - 1.0 / (4.0 * self.ndim_problem)
                + 1.0 / (21.0 * np.square(self.ndim_problem))
            )
        assert (
            self.n_parents > 0
        ), f"`self.n_parents` = {self.n_parents}, but should > 0."
        self.mean = options.get(
            "mean"
        )  # mean of Gaussian search/sampling/mutation distribution
        if self.mean is None:  # `mean` overwrites `x` if both are set
            self.mean = options.get("x")
        # "overall" standard deviation, mutation strength (Nikolaus Hansen, 2023; Hans-Georg Beyer, 2017)
        self.sigma = options.get("sigma")  # global step-size (σ)
        assert self.sigma > 0, f"`self.sigma` = {self.sigma}, but should > 0."
        self.lr_mean = options.get("lr_mean")  # learning rate of mean update
        assert (
            self.lr_mean is None or self.lr_mean > 0
        ), f"`self.lr_mean` = {self.lr_mean}, but should > 0."
        self.lr_sigma = options.get("lr_sigma")  # learning rate of sigma update
        assert (
            self.lr_sigma is None or self.lr_sigma > 0
        ), f"`self.lr_sigma` = {self.lr_sigma}, but should > 0."
        self._printed_evaluations = self.n_function_evaluations
        # set options for *restart*
        self._n_restart = 0  # only for restart
        self._n_generations = 0  # number of generations
        self._list_generations = []  # list of number of generations for all restarts
        self._list_fitness = [self.best_so_far_y]  # only for restart
        self._list_initial_mean = []  # list of mean for each restart
        self.sigma_threshold = options.get(
            "sigma_threshold", 1e-12
        )  # stopping threshold of sigma
        self.stagnation = options.get(
            "stagnation", int(10 + np.ceil(30 * self.ndim_problem / self.n_individuals))
        )
        self.fitness_diff = options.get(
            "fitness_diff", 1e-12
        )  # stopping threshold of fitness difference
        self._sigma_bak = np.copy(
            self.sigma
        )  # initial global step-size -> only for restart

    def _compute_weights(self):
        # unify these following settings in the base class for *consistency* and *simplicity*
        w_base, w = (
            np.log((self.n_individuals + 1.0) / 2.0),
            np.log(np.arange(self.n_parents) + 1.0),
        )
        # positive weight coefficients for weighted intermediate recombination (Nikolaus Hansen, 2023)
        #   [assigning different weights should be interpreted as a selection mechanism]
        w = (w_base - w) / (self.n_parents * w_base - np.sum(w))
        # variance effective selection mass (Nikolaus Hansen, 2023)
        #   effective sample size of the selected samples
        mu_eff = 1.0 / np.sum(np.square(w))  # μ_eff (μ_w)
        return w, mu_eff

    def initialize(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary
            )
        else:
            mean = np.copy(self.mean)
        self.mean = np.copy(mean)
        return mean

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

    def restart_reinitialize(self, y):
        min_y = np.min(y)
        if min_y < self._list_fitness[-1]:
            self._list_fitness.append(min_y)
        else:
            self._list_fitness.append(self._list_fitness[-1])
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (
                self._list_fitness[-self.stagnation] - self._list_fitness[-1]
            ) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if is_restart:
            self._print_verbose_info([], y, True)
            if self.verbose:
                print(" ....... *** restart *** .......")
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self._list_fitness = [np.inf]
            self.sigma = np.copy(self._sigma_bak)
            self.n_individuals *= 2
            self.n_parents = int(self.n_individuals / 2)
            if self.n_parents > 1:
                self._w, self._mu_eff = self._compute_weights()
        return is_restart

    def _collect(self, fitness=None, y=None, mean=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["mean"] = mean  # final mean of search distribution
        results["initial_mean"] = self.mean  # initial mean of search distribution
        results["_list_initial_mean"] = (
            self._list_initial_mean
        )  # list of initial mean for each restart
        results["sigma"] = self.sigma  # only global step-size of search distribution
        results["_n_restart"] = self._n_restart  # number of restart
        results["_n_generations"] = self._n_generations  # number of generations
        results["_list_generations"] = (
            self._list_generations
        )  # list of number of generations for each restart
        return results
