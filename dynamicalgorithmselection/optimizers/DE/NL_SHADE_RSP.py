import numpy as np
from dynamicalgorithmselection.optimizers.DE.DE import DE


class NL_SHADE_RSP(DE):
    start_condition_parameters = ["x", "y", "archive", "MF", "MCr", "k_idx", "pa"]

    def __init__(self, problem, options):
        super().__init__(problem, options)
        self.Nmax = options.get("Nmax", 30 * self.ndim_problem)
        self.Nmin = options.get("Nmin", 4)
        self.n_individuals = self.Nmax

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
        self.memory_size = len(self.MF)
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
        attempts = 0
        while np.any(F <= 0) and attempts < 100:
            idx = np.where(F <= 0)[0]
            F[idx] = self._sample_cauchy(cauchy_locs[idx], 0.1, len(idx))
            attempts += 1
        return Cr, np.minimum(1, F)

    def _update_memory(self, SF, SCr, df):
        if len(SF) > 0:
            w = df / np.sum(df)
            # Weighted Lehmer Mean for F
            self.MF[self.k_idx] = np.sum(w * (SF**2)) / (np.sum(w * SF) + 1e-15)
            # Weighted Arithmetic Mean for Cr
            self.MCr[self.k_idx] = np.sum(w * SCr)
            self.k_idx = (self.k_idx + 1) % self.memory_size

    def iterate(self, x=None, y=None, args=None):
        if x is None or y is None:
            raise ValueError("x and y must be provided for iteration.")
        NP = x.shape[0]

        # Sort population according to fitness for RSP and Crossover mapping
        sort_idx = np.argsort(y)
        x = x[sort_idx]
        y = y[sort_idx]

        Cr, F = self._choose_F_Cr(NP)

        # Sort Cr so better individuals get smaller Cr (for exponential crossover)
        Cr = np.sort(Cr)

        # Adaptive greediness pb (from 0.4 to 0.2)
        nfe_ratio = self.n_function_evaluations / self.max_function_evaluations
        pb = 0.4 - 0.2 * nfe_ratio
        pb_upper = max(2, int(np.round(NP * pb)))

        # Adaptive Cr_b for binomial crossover
        Cr_b = 0.0 if nfe_ratio < 0.5 else 2.0 * (nfe_ratio - 0.5)

        # Rank-based probabilities for r2 (RSP)
        ranks = np.exp(-np.arange(NP) / NP)
        pr = ranks / np.sum(ranks)

        x2 = np.zeros_like(x)
        use_arc = self.rng_optimization.random(NP) < self.pa

        r1 = np.zeros(NP, dtype=int)
        r2 = np.zeros(NP, dtype=int)
        pbest_idx = np.zeros(NP, dtype=int)

        for i in range(NP):
            # pbest index
            valid_pbest = [j for j in range(pb_upper) if j != i]
            pb_i = int(self.rng_optimization.choice(valid_pbest)) if valid_pbest else i
            pbest_idx[i] = pb_i

            # r1 index (uniform)
            valid_r1 = [j for j in range(NP) if j not in (i, pb_i)]
            r1_i = (
                int(self.rng_optimization.choice(valid_r1))
                if valid_r1
                else self.rng_optimization.integers(0, NP)
            )
            r1[i] = r1_i

            # r2 index (archive or RSP)
            if use_arc[i] and len(self.archive) > 0:
                r2[i] = self.rng_optimization.integers(0, len(self.archive))
                x2[i] = self.archive[r2[i]]
            else:
                use_arc[i] = False
                valid_r2 = [j for j in range(NP) if j not in (i, pb_i, r1_i)]

                if valid_r2:
                    # Re-normalize RSP probabilities for the remaining valid choices
                    valid_pr = pr[valid_r2] / np.sum(pr[valid_r2])
                    r2_i = int(self.rng_optimization.choice(valid_r2, p=valid_pr))
                else:
                    r2_i = self.rng_optimization.integers(0, NP)

                r2[i] = r2_i
                x2[i] = x[r2_i]

        # Generate Trials: current-to-pbest/1
        x_pbest = x[pbest_idx]
        vs = x + F[:, np.newaxis] * (x_pbest - x) + F[:, np.newaxis] * (x[r1] - x2)
        vs = np.clip(vs, self.lower_boundary, self.upper_boundary)

        # Dual Crossover Handling
        us = np.copy(x)
        for i in range(NP):
            if self.rng_optimization.random() < 0.5:
                # Binomial crossover with Cr_b
                jrand = self.rng_optimization.integers(self.ndim_problem)
                for j in range(self.ndim_problem):
                    if self.rng_optimization.random() < Cr_b or j == jrand:
                        us[i, j] = vs[i, j]
            else:
                # Exponential crossover with Cr_i
                n1 = self.rng_optimization.integers(self.ndim_problem)
                n2 = 1
                while self.rng_optimization.random() < Cr[i] and n2 < self.ndim_problem:
                    n2 += 1
                for j in range(n2):
                    idx = (n1 + j) % self.ndim_problem
                    us[i, idx] = vs[i, idx]

        # Selection
        new_y = np.array([self._evaluate_fitness(ui, args) for ui in us])
        better_idx = np.where(new_y < y)[0]

        if len(better_idx) > 0:
            # Update Archive Probability (pa)
            df = y[better_idx] - new_y[better_idx]
            arc_used_better = use_arc[better_idx]

            df_A = np.sum(df[arc_used_better])
            df_P = np.sum(df[~arc_used_better])
            n_A_total = np.sum(use_arc)
            n_P_total = NP - n_A_total

            mean_A = df_A / n_A_total if n_A_total > 0 else 0
            mean_P = df_P / n_P_total if n_P_total > 0 else 0

            if mean_A + mean_P > 0:
                self.pa = mean_A / (mean_A + mean_P)
            self.pa = np.clip(self.pa, 0.1, 0.9)  # Clipping rule applied

            # Update Archive
            success_x = x[better_idx]
            self.archive = np.vstack([self.archive, success_x])
            if len(self.archive) > self.NA:
                # Remove random individuals
                remove_idx = self.rng_optimization.choice(
                    len(self.archive), len(self.archive) - self.NA, replace=False
                )
                self.archive = np.delete(self.archive, remove_idx, axis=0)

            # Record successes for memory update
            self._update_memory(F[better_idx], Cr[better_idx], df)

            x[better_idx] = us[better_idx]
            y[better_idx] = new_y[better_idx]

        # NLPSR (Non-Linear Population Size Reduction)
        FEs = self.n_function_evaluations
        MaxFEs = self.max_function_evaluations
        nfe_ratio_nlpsr = FEs / MaxFEs
        new_NP = int(
            np.round(
                (self.Nmin - self.Nmax)
                * np.power(nfe_ratio_nlpsr, 1.0 - nfe_ratio_nlpsr)
                + self.Nmax
            )
        )
        new_NP = max(self.Nmin, new_NP)

        if new_NP < NP:
            sort_idx_final = np.argsort(y)
            x = x[sort_idx_final][:new_NP]
            y = y[sort_idx_final][:new_NP]
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
            old_evals = self.n_function_evaluations
            self._print_verbose_info(fitness, y)
            x, y = self.iterate(x, y, args)
            self.results.update(
                {
                    "x": x,
                    "y": y,
                }
            )
            if self._check_terminations() or self.n_function_evaluations == old_evals:
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
            self.start_conditions = {}
        else:
            indices = np.argsort(y)[: self.n_individuals]
            start_conditions = {}
            start_conditions.update({"x": x[indices], "y": y[indices]})
            self.start_conditions = start_conditions
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
