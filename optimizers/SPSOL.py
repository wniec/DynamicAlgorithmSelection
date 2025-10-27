import time

import numpy as np  # engine for numerical computing

from optimizers.Optimizer import Optimizer


class SPSOL(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if (
            self.n_individuals is None
        ):  # swarm (population) size, aka number of particles
            self.n_individuals = 20
        self.cognition = options.get("cognition", 2.0)  # cognitive learning rate
        assert self.cognition >= 0.0
        self.society = options.get("society", 2.0)  # social learning rate
        assert self.society >= 0.0
        self.max_ratio_v = options.get("max_ratio_v", 0.2)  # maximal ratio of velocity
        assert 0.0 < self.max_ratio_v <= 1.0
        self.is_bound = options.get("is_bound", False)
        self._max_v = self.max_ratio_v * (
            self.upper_boundary - self.lower_boundary
        )  # maximal velocity
        self._min_v = -self._max_v  # minimal velocity
        self._topology = None  # neighbors topology of social learning
        self._n_generations = 0  # initial number of generations
        # set linearly decreasing inertia weights introduced in [Shi&Eberhart, 1998, IEEE-WCCI/CEC]
        self._max_generations = np.ceil(
            self.max_function_evaluations / self.n_individuals
        )
        if self._max_generations == np.inf:
            self._max_generations = 1e2 * self.ndim_problem
        self._w = (
            0.9 - 0.5 * (np.arange(self._max_generations) + 1.0) / self._max_generations
        )  # from 0.9 to 0.4
        self._swarm_shape = (self.n_individuals, self.ndim_problem)
        assert self.n_individuals >= 3  # for ring topology

    def _ring_topology(self, p_x=None, p_y=None, i=None):
        left, right = i - 1, i + 1
        if i == 0:
            left = self.n_individuals - 1
        elif i == self.n_individuals - 1:
            right = 0
        ring = [left, i, right]
        return p_x[ring[int(np.argmin(p_y[ring]))]]

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = self._ring_topology(
                p_x, p_y, i
            )  # online update within ring topology
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = (
                self._w[min(self._n_generations, len(self._w))] * v[i]
                + self.cognition * cognition_rand * (p_x[i] - x[i])
                + self.society * society_rand * (n_x[i] - x[i])
            )  # velocity update
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]  # position update
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)  # fitness evaluation
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]
        self._n_generations += 1
        self.results.update(
            {i: locals()[i] for i in ("v", "x", "y", "p_x", "p_y", "n_x")}
        )
        return v, x, y, p_x, p_y, n_x

    def initialize(
        self, args=None, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None
    ):
        recalculate_y = y is None
        v = (
            self.rng_initialization.uniform(
                self._min_v, self._max_v, size=self._swarm_shape
            )
            if v is None
            else v
        )
        x = (
            x
            if x is not None
            else self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                size=self._swarm_shape,
            )
        )  # positions
        y = y if y is not None else np.empty((self.n_individuals,))  # fitness
        p_x, p_y = (
            p_x if p_x is not None else np.copy(x),
            p_y if p_y is not None else np.copy(y),
        )  # personally previous-best positions and fitness
        n_x = (
            n_x if n_x is not None else np.copy(x)
        )  # neighborly previous-best positions

        if recalculate_y:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y, n_x
                y[i] = self._evaluate_fitness(x[i], args)
            p_y = np.copy(y)
        return v, x, y, p_x, p_y, n_x

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        v = self.start_conditions.get("v", None)
        x = self.start_conditions.get("x", None)
        y = self.start_conditions.get("y", None)
        p_x = self.start_conditions.get("p_x", None)
        p_y = self.start_conditions.get("p_y", None)
        n_x = self.start_conditions.get("n_x", None)

        v, x, y, p_x, p_y, n_x = self.initialize(args, v, x, y, p_x, p_y, n_x)
        while not self.termination_signal:
            self._print_verbose_info(fitness, y)
            v, x, y, p_x, p_y, n_x = self.iterate(v, x, y, p_x, p_y, n_x, args)
        return self._collect(fitness, y)

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
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results

    def set_data(
        self,
        x,
        y,
        v=None,
        p_x=None,
        p_y=None,
        n_x=None,
        best_x=None,
        best_y=None,
        *args,
        **kwargs,
    ):
        start_conditions = {i: None for i in ("x", "y", "v", "p_x", "p_y", "n_x")}
        if x is None or y is None:
            self.start_conditions = start_conditions
            return
        start_conditions["x"] = x
        start_conditions["y"] = y
        if v is None:
            v = self.rng_initialization.uniform(
                self._min_v, self._max_v, size=self._swarm_shape
            )
            random_idx = np.random.randint(self.n_individuals)
            p_x, p_y, n_x = np.copy(x), np.copy(y), np.copy(x)
            p_x[random_idx] = best_x
            p_y[random_idx] = best_y
            n_x[random_idx] = best_x
        start_conditions["v"] = v
        start_conditions["p_x"] = p_x
        start_conditions["n_x"] = n_x
        start_conditions["p_y"] = p_y

        self.start_conditions = start_conditions
