import time

import numpy as np

from optimizers.Optimizer import Optimizer


class LMCMAES(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        # population size, sample size, number of offspring (Nikolaus Hansen, 2023)

        if (
            self.n_individuals is None
        ):  # number of offspring (λ: lambda), offspring population size
            self.n_individuals = 4 + int(
                3 * np.log(self.ndim_problem)
            )  # only for small populations setting
        self.individuals_at_start = self.n_individuals
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
        self.sigma = 1.5  # global step-size (σ)
        self.lr_mean = options.get("lr_mean")  # learning rate of mean update
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
        self.m = options.get(
            "m", 4 + int(3 * np.log(self.ndim_problem))
        )  # number of direction vectors
        self.n_steps = options.get(
            "n_steps", self.m
        )  # target number of generations between vectors
        self.c_c = options.get(
            "c_c", 1.0 / self.m
        )  # learning rate for evolution path update
        self.c_1 = options.get("c_1", 1.0 / (10.0 * np.log(self.ndim_problem + 1.0)))
        self.c_s = options.get(
            "c_s", 0.3
        )  # learning rate for population success rule (PSR)
        self.d_s = options.get("d_s", 1.0)  # damping parameter for PSR
        self.z_star = options.get("z_star", 0.25)  # target success rate for PSR
        self._a = np.sqrt(1.0 - self.c_1)
        self._c = 1.0 / np.sqrt(1.0 - self.c_1)
        self._bd_1 = np.sqrt(1.0 - self.c_1)
        self._bd_2 = self.c_1 / (1.0 - self.c_1)
        self._p_c_1 = 1.0 - self.c_c
        self._p_c_2 = None
        self._j = None
        self._l = None
        self._it = None
        self._rr = None  # for PSR

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

    def restart_reinitialize(
        self,
        mean=None,
        x=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        y=None,
    ):
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
            if self.is_restart:
                mean, x, p_c, s, vm, pm, b, d, y = self.initialize(True)
                self.d_s *= 2.0
        return mean, x, p_c, s, vm, pm, b, d, y

    def _collect(self, fitness=None, y=None, mean=None):
        self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["mean"] = mean  # final mean of search distribution
        results["initial_mean"] = self.mean  # initial mean of search distribution
        results["_list_initial_mean"] = (
            self._list_initial_mean
        )  # list of initial mean for each restart
        # by default, do NOT save covariance matrix of search distribution in order to save memory,
        # owing to its *quadratic* space complexity
        results["sigma"] = self.sigma  # only global step-size of search distribution
        results["_n_restart"] = self._n_restart  # number of restart
        results["_n_generations"] = self._n_generations  # number of generations
        results["_list_generations"] = (
            self._list_generations
        )  # list of number of generations for each restart
        return results

    def initialize(
        self,
        is_restart=False,
        mean=None,
        x=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        y=None,
    ):
        mean = (
            mean if mean is not None else self._initialize_mean(is_restart)
        )  # mean of Gaussian search distribution
        x = (
            x if x is not None else np.empty((self.n_individuals, self.ndim_problem))
        )  # offspring population
        p_c = (
            p_c if p_c is not None else np.zeros((self.ndim_problem,))
        )  # evolution path
        s = s if s is not None else 0.0  # for PSR of global step-size adaptation
        vm = vm if vm is not None else np.empty((self.m, self.ndim_problem))
        pm = pm if pm is not None else np.empty((self.m, self.ndim_problem))
        b = b if b is not None else np.empty((self.m,))
        d = d if d is not None else np.empty((self.m,))
        y = (
            y if y is not None else np.empty((self.n_individuals,))
        )  # fitness (no evaluation)
        self._p_c_2 = np.sqrt(self.c_c * (2.0 - self.c_c) * self._mu_eff)
        self._rr = np.arange(self.n_individuals * 2, 0, -1) - 1
        self._j = [None] * self.m
        self._l = [None] * self.m
        self._it = 0
        return mean, x, p_c, s, vm, pm, b, d, y

    def _a_z(self, z=None, pm=None, vm=None, b=None):  # Algorithm 3 Az()
        x = np.copy(z)
        for t in range(self._it):
            x = self._a * x + b[self._j[t]] * np.dot(vm[self._j[t]], z) * pm[self._j[t]]
        return x

    def iterate(self, mean=None, x=None, pm=None, vm=None, y=None, b=None, args=None):
        sign, a_z = 1, np.empty((self.ndim_problem,))  # for mirrored sampling
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            if sign == 1:
                z = self.rng_optimization.standard_normal((self.ndim_problem,))
                a_z = self._a_z(z, pm, vm, b)
            x[k] = mean + sign * self.sigma * a_z
            y[k] = self._evaluate_fitness(x[k], args)
            sign *= -1  # sampling in the opposite direction for mirrored sampling
        return x, y

    def _a_inv_z(self, v=None, vm=None, d=None, i=None):  # Algorithm 4 Ainvz()
        x = np.copy(v)
        for t in range(0, i):
            x = self._c * x - d[self._j[t]] * np.dot(vm[self._j[t]], x) * vm[self._j[t]]
        return x

    def _update_distribution(
        self,
        mean=None,
        x=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        y=None,
        y_bak=None,
    ):
        mean_bak = np.dot(self._w, x[np.argsort(y)[: self.n_parents]])
        p_c = self._p_c_1 * p_c + self._p_c_2 * (mean_bak - mean) / self.sigma
        i_min = 1
        if self._n_generations < self.m:
            self._j[self._n_generations] = self._n_generations
        else:
            d_min = self._l[self._j[i_min]] - self._l[self._j[i_min - 1]]
            for j in range(2, self.m):
                d_cur = self._l[self._j[j]] - self._l[self._j[j - 1]]
                if d_cur < d_min:
                    d_min, i_min = d_cur, j
            # start from 0 if all pairwise distances exceed `self.n_steps`
            i_min = 0 if d_min >= self.n_steps else i_min
            # update indexes of evolution paths (`self._j[i_min]` is index of evolution path needed to delete)
            updated = self._j[i_min]
            for j in range(i_min, self.m - 1):
                self._j[j] = self._j[j + 1]
            self._j[self.m - 1] = updated
        self._it = np.minimum(self._n_generations + 1, self.m)
        self._l[self._j[self._it - 1]] = self._n_generations  # to update its generation
        pm[self._j[self._it - 1]] = p_c  # to add the latest evolution path
        # since `self._j[i_min]` is deleted, all vectors (from vm) depending on it need to be computed again
        for i in range(0 if i_min == 1 else i_min, self._it):
            vm[self._j[i]] = self._a_inv_z(pm[self._j[i]], vm, d, i)
            v_n = np.dot(vm[self._j[i]], vm[self._j[i]])
            bd_3 = np.sqrt(1.0 + self._bd_2 * v_n)
            b[self._j[i]] = self._bd_1 / v_n * (bd_3 - 1.0)
            d[self._j[i]] = 1.0 / (self._bd_1 * v_n) * (1.0 - 1.0 / bd_3)
        if self._n_generations > 0:  # for population success rule (PSR)
            r = np.argsort(np.hstack((y, y_bak)))
            z_psr = np.sum(
                self._rr[r < self.n_individuals] - self._rr[r >= self.n_individuals]
            )
            z_psr = z_psr / np.power(self.n_individuals, 2) - self.z_star
            s = (1.0 - self.c_s) * s + self.c_s * z_psr
            self.sigma *= np.exp(s / self.d_s)
        return mean_bak, p_c, s, vm, pm, b, d

    def optimize(
        self, fitness_function=None, args=None
    ):  # for all generations (iterations)
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        mean = self.start_conditions.get("mean", None)
        x = self.start_conditions.get("x", None)
        p_c = self.start_conditions.get("p_c", None)
        s = self.start_conditions.get("s", None)
        vm = self.start_conditions.get("vm", None)
        pm = self.start_conditions.get("pm", None)
        b = self.start_conditions.get("b", None)
        d = self.start_conditions.get("d", None)
        y = self.start_conditions.get("y", None)
        mean, x, p_c, s, vm, pm, b, d, y = self.initialize(
            args, mean, x, p_c, s, vm, pm, b, d, y
        )
        while not self.termination_signal:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(mean, x, pm, vm, y, b, args)

            mean, p_c, s, vm, pm, b, d = self._update_distribution(
                mean, x, p_c, s, vm, pm, b, d, y, y_bak
            )
            self.results.update(
                {
                    i: locals()[i]
                    for i in ("p_c", "s", "vm", "pm", "b", "d", "x", "y", "mean")
                }
            )
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, x, p_c, s, vm, pm, b, d, y = self.restart_reinitialize(
                mean, x, p_c, s, vm, pm, b, d, y
            )

        results = self._collect(fitness, y, mean)
        results["p_c"] = p_c
        results["s"] = s
        return results

    def set_data(
        self,
        x,
        y,
        mean=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        *args,
        **kwargs,
    ):
        if x is None or y is None:
            self.start_conditions = {"x": None, "y": None, "mean": None}
        else:
            indices = np.argsort(y)[: self.individuals_at_start]
            start_conditions = {
                i: locals()[i] for i in ("p_c", "s", "vm", "pm", "b", "d")
            }
            start_conditions.update(
                {
                    "x": x[indices],
                    "y": y[indices],
                    "mean": (x.mean(axis=0) if x is not None else None),
                }
            )
            self.start_conditions = start_conditions
