import numpy as np
from dynamicalgorithmselection.optimizers.DE.DE import DE


class NL_SHADE_RSP(DE):
    start_condition_parameters = ["x", "y", "archive", "MF", "MCr", "k_idx"]

    def __init__(self, problem, options):
        super().__init__(problem, options)
        self.Nmax = self.n_individuals if self.n_individuals else 170
        self.Nmin = options.get("Nmin", 30)
        self.n_individuals = self.Nmax

        self.pb = 0.4
        self.pa = 0.5

        # Archive
        self.NA = int(self.Nmax * 2.1)
        self.archive = np.empty((0, self.ndim_problem))

        # Memory MF and MCr
        self.memory_size = self.ndim_problem * 20
        self.MF = np.ones(self.memory_size) * 0.2
        self.MCr = np.ones(self.memory_size) * 0.2
        self.k_idx = 0

    def initialize(self, args=None, x=None, y=None):
        if x is None:
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                (self.n_individuals, self.ndim_problem),
            )
        if y is None:
            y = np.array([self._evaluate_fitness(xi, args) for xi in x])
        return x, y

    def _sample_cauchy(self, loc, scale, size):
        """Manual Cauchy sampling: loc + scale * tan(pi * (rand - 0.5))"""
        rand = self.rng_optimization.random(size)
        return loc + scale * np.tan(np.pi * (rand - 0.5))

    def _choose_F_Cr(self, NP):
        ind_r = self.rng_optimization.integers(0, self.memory_size, size=NP)
        # Crossover Rate (Normal)
        Cr = self.rng_optimization.normal(loc=self.MCr[ind_r], scale=0.1, size=NP)
        Cr = np.clip(Cr, 0, 1)
        # Step Length (Cauchy)
        cauchy_locs = self.MF[ind_r]
        F = self._sample_cauchy(cauchy_locs, 0.1, NP)
        # Symmetry correction for negative values
        while np.any(F <= 0):
            idx = np.where(F <= 0)[0]
            F[idx] = self._sample_cauchy(cauchy_locs[idx], 0.1, len(idx))
        return Cr, np.minimum(1, F)

    def _update_memory(self, SF, SCr, df):
        if len(SF) > 0:
            w = df / np.sum(df)
            # Weighted Lehmer Mean for F
            self.MF[self.k_idx] = np.sum(w * (SF**2)) / (np.sum(w * SF) + 1e-15)
            # Weighted Arithmetic Mean for Cr
            self.MCr[self.k_idx] = np.sum(w * SCr)
            self.k_idx = (self.k_idx + 1) % self.memory_size

    def iterate(self, x, y, args=None):
        NP = x.shape[0]
        Cr, F = self._choose_F_Cr(NP)

        # Mutation: current-to-pbest/1 with archive
        pb_upper = int(max(2, NP * self.pb))
        pbest_idx = np.argsort(y)[:pb_upper]
        x_pbest = x[self.rng_optimization.choice(pbest_idx, NP)]

        r1 = self.rng_optimization.integers(0, NP, size=NP)
        # Ensure distinct r1
        for i in range(NP):
            while r1[i] == i:
                r1[i] = self.rng_optimization.integers(0, NP)

        # Archive vs Population selection for x2
        x2 = np.zeros_like(x)
        use_arc = self.rng_optimization.random(NP) < self.pa
        arc_idx = np.where(use_arc & (len(self.archive) > 0))[0]
        pop_idx = np.where(~use_arc | (len(self.archive) == 0))[0]

        if len(pop_idx) > 0:
            r2 = self.rng_optimization.integers(0, NP, size=len(pop_idx))
            x2[pop_idx] = x[r2]
        if len(arc_idx) > 0:
            r_arc = self.rng_optimization.integers(
                0, len(self.archive), size=len(arc_idx)
            )
            x2[arc_idx] = self.archive[r_arc]

        # Generate Trials
        vs = x + F[:, np.newaxis] * (x_pbest - x) + F[:, np.newaxis] * (x[r1] - x2)
        vs = np.clip(vs, self.lower_boundary, self.upper_boundary)

        # Binomial Crossover
        jrand = self.rng_optimization.integers(self.ndim_problem, size=NP)
        mask = self.rng_optimization.random((NP, self.ndim_problem)) < Cr[:, np.newaxis]
        us = np.where(mask, vs, x)
        us[np.arange(NP), jrand] = vs[np.arange(NP), jrand]

        # Selection
        new_y = np.array([self._evaluate_fitness(ui, args) for ui in us])
        better = new_y < y

        if np.any(better):
            # Update Archive
            success_x = x[better]
            self.archive = np.vstack([self.archive, success_x])
            if len(self.archive) > self.NA:
                self.archive = self.archive[-self.NA :]

            # Record successes for memory
            df = (y[better] - new_y[better]) / (y[better] + 1e-15)
            self._update_memory(F[better], Cr[better], df)

            x[better], y[better] = us[better], new_y[better]

        # NLPSR
        FEs, MaxFEs = self.n_function_evaluations, self.max_function_evaluations
        new_NP = int(
            np.round(
                self.Nmax
                + (self.Nmin - self.Nmax) * np.power(FEs / MaxFEs, 1 - FEs / MaxFEs)
            )
        )
        if new_NP < NP:
            idx = np.argsort(y)[:new_NP]
            x, y = x[idx], y[idx]
            self.n_individuals = new_NP
            self.NA = int(max(new_NP * 2.1, self.Nmin))

        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = DE.optimize(self, fitness_function)

        x = self.start_conditions.get("x", None)
        y = self.start_conditions.get("y", None)

        x, y = self.initialize(args, x, y)

        while True:
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, args)
            self.results.update(
                {
                    "x": x,
                    "y": y,
                }
            )
            if self._check_terminations():
                break

        return self._collect(fitness, y)

    def set_data(
        self,
        x=None,
        y=None,
        *args,
        **kwargs,
    ):
        if x is None or y is None:
            self.start_conditions = {"x": None, "y": None}
        elif not isinstance(y, np.ndarray):
            loc = locals()
            self.start_conditions = {}
        else:
            indices = np.argsort(y)[: self.n_individuals]
            start_conditions = {}
            start_conditions.update({"x": x[indices], "y": y[indices]})
            self.start_conditions = start_conditions
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
