import numpy as np
from dynamicalgorithmselection.optimizers.DE.DE import DE


class JDE21(DE):
    """Implementation adapted to exactly mirror the discrepancies found in the provided
    optimizer.py and Population.py, including continuous NLPSR and unused success archives."""

    start_condition_parameters = ["x", "y", "F", "Cr"]

    def __init__(self, problem, options):
        super().__init__(problem, options)
        # Population parameters
        self.bNP = 160
        self.sNP = 10
        self.n_individuals = self.bNP + self.sNP

        # Stagnation and Reset parameters
        self.age = 0
        self.eps = 1e-12
        self.MyEps = 0.25

        # Self-adaptation probabilities
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.Finit = 0.5
        self.CRinit = 0.9

        # Parameter Limits (Big Population)
        self.Fl_b = 0.1
        self.CRl_b = 0.0
        self.CRu_b = 1.1

        # Parameter Limits (Small Population)
        self.Fl_s = 0.17
        self.CRl_s = 0.1
        self.CRu_s = 0.8  # Note: ignored in optimizer.py logic

        # Shared Upper Bound for F
        self.Fu = 1.1

        self.F = np.full(self.n_individuals, self.Finit)
        self.Cr = np.full(self.n_individuals, self.CRinit)

    def initialize(self, args=None, x=None, y=None):
        if x is None:
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                (self.n_individuals, self.ndim_problem),
            )
        else:
            self.n_individuals = x.shape[0]
            self.bNP = self.n_individuals - self.sNP
        if y is None:
            y = np.array(
                [
                    self._evaluate_fitness(xi, args, F=self.F[i], Cr=self.Cr[i])
                    for i, xi in enumerate(x)
                ]
            )

        # Discrepancy: Initialize dead history archives from Population.py
        self.MF = np.ones(self.ndim_problem * 20) * 0.2
        self.MCr = np.ones(self.ndim_problem * 20) * 0.2
        self.k = 0

        return x, y

    def _reflect_bounds(self, v):
        return np.clip(v, self.initial_lower_boundary, self.initial_upper_boundary)

    def _check_population_reduction(self, x, y):
        # SYNCHRONIZATION
        actual_size = len(y)
        if actual_size != self.n_individuals:
            self.n_individuals = actual_size
            self.bNP = self.n_individuals - self.sNP

            if len(self.F) != actual_size:
                self.F = np.full(actual_size, self.Finit)
                self.Cr = np.full(actual_size, self.CRinit)

        # Discrepancy: Continuous NLPSR logic from Population.py
        progress = self.n_function_evaluations / self.max_function_evaluations
        if progress >= 1.0:
            return x, y

        new_NP = int(
            np.round(
                self.Nmax + (self.Nmin - self.Nmax) * np.power(progress, 1 - progress)
            )
        )

        if new_NP < self.n_individuals:
            # Discrepancy: optimizer.py simply takes the last NP elements
            x = x[-new_NP:]
            y = y[-new_NP:]
            self.F = self.F[-new_NP:]
            self.Cr = self.Cr[-new_NP:]
            self.n_individuals = new_NP
            self.bNP = new_NP - self.sNP

        return x, y

    def r_choice(self, preferred_pool, exclude):
        valid = [idx for idx in preferred_pool if idx not in exclude]
        return self.rng_optimization.choice(valid) if valid else exclude[0]

    def _evolve_population(self, x, y, args, is_big=True):
        if self.n_individuals == 0:
            return x, y, [], [], []

        start_idx = 0 if is_big else self.bNP
        end_idx = self.bNP if is_big else self.n_individuals

        f_low = self.Fl_b if is_big else self.Fl_s
        # Discrepancy: optimizer.py ignores CRu_s and uses CRu_b for both!
        cr_bound = self.CRu_b
        cr_low = self.CRl_b if is_big else self.CRl_s

        SF, SCr, df = [], [], []

        # Snapshot population so all mutations/crossovers/crowding reference the same state
        x_snapshot = x.copy()

        for i in range(start_idx, end_idx):
            # Parameter Adaptation
            new_F = (
                self.rng_optimization.random() * self.Fu + f_low
                if self.rng_optimization.random() < self.tau1
                else self.F[i]
            )
            new_Cr = (
                self.rng_optimization.random() * cr_bound + cr_low
                if self.rng_optimization.random() < self.tau2
                else self.Cr[i]
            )

            # Mutation Pool Selection
            if is_big:
                progress = self.n_function_evaluations / self.max_function_evaluations
                ms_size = 1 if progress <= 1 / 3 else 2 if progress <= 2 / 3 else 3

                available_sNP = self.n_individuals - self.bNP
                ms_size = min(ms_size, available_sNP)

                if ms_size > 0:
                    ms_indices = self.rng_optimization.choice(
                        range(self.bNP, self.n_individuals), ms_size, replace=False
                    )
                else:
                    ms_indices = np.array([], dtype=int)

                pool_r2_r3 = np.concatenate([np.arange(self.bNP), ms_indices])

                r1 = self.r_choice(range(self.bNP), [i])
                r2 = self.r_choice(pool_r2_r3, [i, r1])
                r3 = self.r_choice(pool_r2_r3, [i, r1, r2])

            else:
                pool = [idx for idx in range(self.bNP, self.n_individuals) if idx != i]

                if len(pool) >= 3:
                    r1, r2, r3 = self.rng_optimization.choice(pool, 3, replace=False)
                else:
                    full_pool = [idx for idx in range(self.n_individuals) if idx != i]
                    if len(full_pool) >= 3:
                        r1, r2, r3 = self.rng_optimization.choice(
                            full_pool, 3, replace=False
                        )
                    else:
                        full_pool_with_i = list(range(self.n_individuals))
                        r1, r2, r3 = self.rng_optimization.choice(
                            full_pool_with_i, 3, replace=True
                        )

            # Mutation and Reflection (use snapshot so all mutations reference pre-update state)
            v = x_snapshot[r1] + new_F * (x_snapshot[r2] - x_snapshot[r3])
            v = self._reflect_bounds(v)

            # Crossover (Rotational Invariant Strategy)
            if new_Cr > 1.0:
                u = v.copy()
            else:
                u = x_snapshot[i].copy()
                j_rand = self.rng_optimization.integers(0, self.ndim_problem)
                mask = self.rng_optimization.random(self.ndim_problem) <= new_Cr
                mask[j_rand] = True
                u[mask] = v[mask]

            # Evaluate
            new_y = self._evaluate_fitness(u, args, F=self.F[i], Cr=self.Cr[i])

            # Crowding & Selection
            if is_big:
                dists = np.sum((x_snapshot[: self.bNP] - u) ** 2, axis=1)
                target = np.argmin(dists)
            else:
                target = i

            if new_y <= y[target]:
                # Track for unused history archives
                SF.append(new_F)
                SCr.append(new_Cr)
                d = (y[target] - new_y) / (y[target] + 1e-9)
                df.append(d)

                x[target], y[target] = u, new_y
                self.F[target], self.Cr[target] = new_F, new_Cr

                if is_big and new_y < self.best_so_far_y:
                    self.best_so_far_y = new_y
                    self.age = 0
            elif is_big and target == i:
                self.age += 1

        return x, y, SF, SCr, df

    def iterate(self, x=None, y=None, args=None):
        x, y = self._check_population_reduction(x, y)

        # P_b Reinitialization Check
        if self.bNP > 0:
            best_b_y = np.min(y[: self.bNP])
            # Discrepancy: prevecEnakih logic
            eqs_b = np.sum(np.abs(y[: self.bNP] - best_b_y) < self.eps)
            age_limit = 0.1 * self.max_function_evaluations

            if (eqs_b > 2 and eqs_b > self.bNP * self.MyEps) or (self.age > age_limit):
                x[: self.bNP] = self.rng_initialization.uniform(
                    self.initial_lower_boundary,
                    self.initial_upper_boundary,
                    (self.bNP, self.ndim_problem),
                )
                self.F[: self.bNP] = self.Finit
                self.Cr[: self.bNP] = self.CRinit
                # Discrepancy: Setting cost explicitly to 1e15 without evaluating
                y[: self.bNP] = 1e15
                self.age = 0

        # P_s Reinitialization Check
        if self.sNP > 0:
            best_s_idx = self.bNP + np.argmin(y[self.bNP :])
            eqs_s = np.sum(np.abs(y[self.bNP :] - y[best_s_idx]) < self.eps)

            if eqs_s > 2 and eqs_s > self.sNP * self.MyEps:
                best_x_s = x[best_s_idx].copy()
                best_y_s = y[best_s_idx]

                x[self.bNP :] = self.rng_initialization.uniform(
                    self.initial_lower_boundary,
                    self.initial_upper_boundary,
                    (self.sNP, self.ndim_problem),
                )
                self.F[self.bNP :] = self.Finit
                self.Cr[self.bNP :] = self.CRinit
                # Discrepancy: Setting cost explicitly to 1e15
                y[self.bNP :] = 1e15

                x[best_s_idx] = best_x_s
                y[best_s_idx] = best_y_s

        SF_total, SCr_total, df_total = [], [], []

        # Big Population Generation
        if self.bNP > 0:
            x, y, SF, SCr, df = self._evolve_population(x, y, args, is_big=True)
            SF_total.extend(SF)
            SCr_total.extend(SCr)
            df_total.extend(df)

        # Migration
        if self.bNP > 0 and self.sNP > 0:
            best_overall_idx = np.argmin(y)
            if best_overall_idx < self.bNP:
                # Discrepancy: Overwrites explicitly the first index of P_s (self.bNP)
                x[self.bNP] = x[best_overall_idx].copy()
                y[self.bNP] = y[best_overall_idx]

        # Small Population Generation (repeats m times)
        if self.sNP > 0:
            m = self.bNP // self.sNP if self.bNP > 0 else 1
            m = max(1, m)
            for _ in range(m):
                x, y, SF, SCr, df = self._evolve_population(x, y, args, is_big=False)
                SF_total.extend(SF)
                SCr_total.extend(SCr)
                df_total.extend(df)

        # Discrepancy: Update dead history archives
        if len(SF_total) > 0:
            SF_arr = np.array(SF_total)
            SCr_arr = np.array(SCr_total)
            df_arr = np.array(df_total)

            def mean_wL(df_vals, s_vals):
                w = df_vals / np.sum(df_vals)
                if np.sum(w * s_vals) > 0.000001:
                    return np.sum(w * (s_vals**2)) / np.sum(w * s_vals)
                else:
                    return 0.5

            self.MF[self.k] = mean_wL(df_arr, SF_arr)
            self.MCr[self.k] = mean_wL(df_arr, SCr_arr)
            self.k = (self.k + 1) % len(self.MF)
        else:
            self.MF[self.k] = 0.5
            self.MCr[self.k] = 0.5

        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        x, y = self.initialize(
            args, self.start_conditions.get("x"), self.start_conditions.get("y")
        )

        self.best_so_far_y = np.min(y)

        while True:
            old_evals = self.n_function_evaluations
            x, y = self.iterate(x, y, args)
            self.results.update({"x": x, "y": y, "Cr": self.Cr[:], "F": self.F[:]})
            if self._check_terminations() or self.n_function_evaluations == old_evals:
                break

        return self._collect(fitness, y)

    def set_data(self, x=None, y=None, *args, **kwargs):
        if x is None or y is None:
            self.start_conditions = {"x": None, "y": None}
        elif not isinstance(y, np.ndarray):
            self.start_conditions = {}
        else:
            indices = np.argsort(y)[: self.n_individuals]
            self.start_conditions = {"x": x[indices], "y": y[indices]}
            Cr = kwargs.get("Cr")
            if Cr is not None:
                self.Cr = Cr[indices]
            F = kwargs.get("F")
            if F is not None:
                self.F = F[indices]
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
