import numpy as np
from dynamicalgorithmselection.optimizers.DE.DE import DE


class NL_SHADE_RSP(DE):
    """Implementation of this algorithm tries to be faithful both to original paper
    and to its implementation in RL-DAS project.
    In case of any difference, it follows RL-DAS approach"""

    start_condition_parameters = ["x", "y", "archive", "MF", "MCr", "k_idx", "pa"]

    def __init__(self, problem, options):
        super().__init__(problem, options)
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
            y = np.array(
                [
                    self._evaluate_fitness(
                        xi,
                        args,
                    )
                    for xi in x
                ]
            )
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
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]

        return Cr, np.minimum(1, F)

    def _update_memory(self, SF, SCr, df):
        if len(SF) > 0:
            w = df / np.sum(df)

            sum_w_SF = np.sum(w * SF)
            if sum_w_SF > 0.000001:
                mean_wL_F = np.sum(w * (SF**2)) / sum_w_SF
            else:
                mean_wL_F = 0.5

            sum_w_SCr = np.sum(w * SCr)
            if sum_w_SCr > 0.000001:
                mean_wL_Cr = np.sum(w * (SCr**2)) / sum_w_SCr
            else:
                mean_wL_Cr = 0.5

            self.MF[self.k_idx] = mean_wL_F
            self.MCr[self.k_idx] = mean_wL_Cr
            self.k_idx = (self.k_idx + 1) % self.memory_size
        else:
            self.MF[self.k_idx] = 0.5
            self.MCr[self.k_idx] = 0.5
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

        # Sort Cr so better individuals get smaller Cr
        Cr = np.sort(Cr)

        # Adaptive greediness pb (from 0.4 to 0.2)
        nfe_ratio = self.n_function_evaluations / self.max_function_evaluations
        pb = 0.4 - 0.2 * nfe_ratio
        pb_upper = max(2, int(np.round(NP * pb)))

        # BUG : Inverted Cr_b calculation - same as in RL-DAS implementation, did so for compatibility of comparison
        # It is non-zero (negative) in the first half and 0.0 in the second half
        Cr_b = 2.0 * (nfe_ratio - 0.5) if nfe_ratio < 0.5 else 0.0

        # Rank-based probabilities for r2 (RSP)
        ranks = np.exp(-(np.arange(NP) + 1) / NP)
        pr = ranks / np.sum(ranks)

        x2 = np.zeros_like(x)
        use_arc = self.rng_optimization.random(NP) < self.pa

        # Logical Change 3: Archive indices canceled if archive is too small
        if len(self.archive) < 25:
            use_arc[:] = False

        r1 = np.zeros(NP, dtype=int)
        r2 = np.zeros(NP, dtype=int)
        pbest_idx = np.zeros(NP, dtype=int)

        for i in range(NP):
            # Logical Change 3: pbest index with bounded retries
            pb_i = self.rng_optimization.integers(0, pb_upper)
            count = 0
            while pb_i == i and count < 1:
                pb_i = self.rng_optimization.integers(0, NP)
                count += 1
            pbest_idx[i] = pb_i

            # Logical Change 3: r1 index with bounded retries
            r1_i = self.rng_optimization.integers(0, NP)
            count = 0
            while (r1_i == i or r1_i == pb_i) and count < 25:
                r1_i = self.rng_optimization.integers(0, NP)
                count += 1
            r1[i] = r1_i

            # Logical Change 3: r2 index (archive or RSP) with bounded retries
            if use_arc[i] and len(self.archive) > 0:
                r2[i] = self.rng_optimization.integers(
                    0, min(len(self.archive), self.NA)
                )
                x2[i] = self.archive[r2[i]]
            else:
                r2_i = int(self.rng_optimization.choice(np.arange(NP), p=pr))
                count = 0
                while (r2_i == i or r2_i == pb_i or r2_i == r1_i) and count < 25:
                    r2_i = int(self.rng_optimization.choice(np.arange(NP), p=pr))
                    count += 1
                r2[i] = r2_i
                x2[i] = x[r2_i]

        # Generate Trials: current-to-pbest/1
        x_pbest = x[pbest_idx]
        vs = x + F[:, np.newaxis] * (x_pbest - x) + F[:, np.newaxis] * (x[r1] - x2)

        # Note: Removed the correct np.clip() here to implement Bug 5

        us = np.copy(x)

        CrossExponential = self.rng_optimization.random() < 0.5

        # ^ Bug copied from RL-DAS implementation
        if CrossExponential:
            # Executes Binomial logic with Cr_b when CrossExponential is True -> RL-DAS bug compatibility
            for i in range(NP):
                jrand = self.rng_optimization.integers(self.ndim_problem)
                for j in range(self.ndim_problem):
                    if self.rng_optimization.random() < Cr_b or j == jrand:
                        us[i, j] = vs[i, j]
        else:
            # Executes Exponential logic with Cr when CrossExponential is False -> RL-DAS bug compatibility
            for i in range(NP):
                n1 = self.rng_optimization.integers(self.ndim_problem)
                n2 = 1
                while self.rng_optimization.random() < Cr[i] and n2 < self.ndim_problem:
                    n2 += 1
                for j in range(n2):
                    idx = (n1 + j) % self.ndim_problem
                    us[i, idx] = vs[i, idx]

        # BUG 5: Hardcoded [-100, 100] bounds
        out_of_bounds = (us < -100) | (us > 100)
        if np.any(out_of_bounds):
            us = np.where(
                out_of_bounds,
                self.rng_optimization.uniform(-100, 100, size=us.shape),
                us,
            )

        # Selection

        new_y = np.array(
            [
                self._evaluate_fitness(
                    ui,
                    args,
                )
                for ui in us
            ]
        )
        better_idx = np.where(new_y < y)[0]

        if len(better_idx) > 0:
            # Logical Change 4: Normalized df calculation
            df = (y[better_idx] - new_y[better_idx]) / (y[better_idx] + 1e-9)

            arc_used_better = use_arc[better_idx]

            fp = np.sum(df[arc_used_better])
            fa = np.sum(df[~arc_used_better])
            na = np.sum(arc_used_better)

            if na == 0 or fa == 0:
                self.pa = 0.5
            else:
                self.pa = (fa / (na + 1e-15)) / (
                    (fa / (na + 1e-15)) + (fp / (NP - na + 1e-15))
                )
                self.pa = np.clip(self.pa, 0.1, 0.9)

            # Logical Change 6: One-by-one Archive updating/trimming
            for i in better_idx:
                if len(self.archive) < self.NA:
                    self.archive = np.vstack([self.archive, x[i]])
                else:
                    replace_idx = self.rng_optimization.integers(0, len(self.archive))
                    self.archive[replace_idx] = x[i]

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
            # Slice archive if it exceeds the new reduced NA
            if len(self.archive) > self.NA:
                self.archive = self.archive[: self.NA]

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
                    "archive": self.archive[:],
                    "MF": self.MF[:],
                    "MCr": self.MCr[:],
                    "k_idx": self.k_idx,
                    "pa": self.pa,
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
            for var in ["archive", "MF", "MCr", "k_idx", "pa"]:
                if var in kwargs:
                    setattr(self, var, kwargs[var])
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
