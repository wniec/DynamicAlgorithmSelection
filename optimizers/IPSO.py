import numpy as np  # engine for numerical computing
from optimizers.Optimizer import Optimizer


class IPSO(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.results: dict = {}
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
        self.n_individuals = 1  # minimum of swarm size
        self.max_n_individuals = options.get(
            "max_n_individuals", 1000
        )  # maximum of swarm size
        assert self.max_n_individuals > 0
        self.cognition = options.get("cognition", 2.05)  # cognitive learning rate
        assert self.cognition > 0.0
        self.society = options.get("society", 2.05)  # social learning rate
        assert self.society > 0.0
        self.constriction = options.get("constriction", 0.729)  # constriction factor
        assert self.constriction > 0.0
        self.max_ratio_v = options.get("max_ratio_v", 0.5)  # maximal ratio of velocity
        assert 0.0 <= self.max_ratio_v <= 1.0

    def initialize(self, args=None, v=None, x=None, y=None, p_x=None, p_y=None):
        recalculate_y = y is None
        v = (
            np.zeros((self.n_individuals, self.ndim_problem)) if v is None else v
        )  # velocities
        x = (
            self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                size=self._swarm_shape,
            )
            if x is None
            else x
        )  # positions
        y = np.empty((self.n_individuals,)) if y is None else y  # fitness
        p_x, p_y = (
            (np.copy(x) if p_x is None else p_x),
            (np.copy(y) if p_y is None else p_y),
        )
        # personally previous-best positions and fitness
        if recalculate_y:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y
                y[i] = self._evaluate_fitness(x[i], args)
            p_y = np.copy(y)
        return v, x, y, p_x, p_y

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

    def iterate(
        self, v=None, x=None, y=None, p_x=None, p_y=None, args=None, fitness=None
    ):
        for i in range(self.n_individuals):  # horizontal social learning
            if self._check_terminations():
                return v, x, y, p_x, p_y
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = self.constriction * (
                v[i]
                + self.cognition * cognition_rand * (p_x[i] - x[i])
                + self.society * society_rand * (p_x[np.argmin(p_y)] - x[i])
            )  # velocity update
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]  # position update
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]
        if (
            self.n_individuals < self.max_n_individuals
        ):  # population growth (vertical social learning)
            if self._check_terminations():
                return v, x, y, p_x, p_y
            xx = self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)
            model = p_x[np.argmin(p_y)]  # the best particle is used as model
            # use different random numbers of different dimensions for diversity (important),
            # which is *slightly different* from the original paper but often with better performance
            # xx += self.rng_optimization.uniform()*(model - xx)  # from the original paper
            xx += self.rng_optimization.uniform(size=(self.ndim_problem,)) * (
                model - xx
            )
            xx = np.clip(xx, self.lower_boundary, self.upper_boundary)
            yy = self._evaluate_fitness(xx, args)
            v = np.vstack((v, np.zeros((self.ndim_problem,))))
            x, y = np.vstack((x, xx)), np.hstack((y, yy))
            p_x, p_y = np.vstack((p_x, xx)), np.hstack((p_y, yy))
            self.n_individuals += 1
        self._n_generations += 1
        self.results.update({i: locals()[i] for i in ("v", "x", "y", "p_x", "p_y")})
        return v, x, y, p_x, p_y

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)

        v = self.start_conditions.get("v", None)
        x = self.start_conditions.get("x", None)
        y = self.start_conditions.get("y", None)
        p_x = self.start_conditions.get("p_x", None)
        p_y = self.start_conditions.get("p_y", None)

        v, x, y, p_x, p_y = self.initialize(args, v, x, y, p_x, p_y)
        while not self.termination_signal:
            self._print_verbose_info(fitness, y)
            v, x, y, p_x, p_y = self.iterate(v, x, y, p_x, p_y, args)
        return self._collect(fitness, y)

    def set_data(
        self,
        x,
        y,
        v=None,
        p_x=None,
        p_y=None,
        best_x=None,
        best_y=None,
        *args,
        **kwargs,
    ):
        start_conditions = {i: None for i in ("x", "y", "v", "p_x", "p_y")}
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
            p_x, p_y = np.copy(x), np.copy(y)
            p_x[random_idx] = best_x
            p_y[random_idx] = best_y
        start_conditions["v"] = v
        start_conditions["p_x"] = p_x
        start_conditions["p_y"] = p_y

        self.start_conditions = start_conditions

    def get_data(self):
        pop_data = ["x", "v", "y", "p_x", "p_y"]
        if "y" in self.results:
            best_indices = sorted(
                [i for i in range(self.n_individuals)],
                key=lambda x: self.results["y"][x],
            )[: self.n_individuals]
        return {
            k: (v[best_indices] if k in pop_data else v)
            for k, v in self.results.items()
        } or self.start_conditions
