import numpy as np
from dynamicalgorithmselection.optimizers.DE.DE import DE


class JDE21(DE):
    start_condition_parameters = ["x", "y", "F", "Cr"]

    def __init__(self, problem, options):
        super().__init__(problem, options)

        # Mathematical minimum population limit to survive RL starvation
        self.Nmin = 4

        # Population parameters
        # We start with the base sizes defined in the j21 paper,
        # though set_data/initialize will override this if the RL agent injects a different size.
        self.bNP = 160
        self.sNP = 10
        self.n_individuals = self.bNP + self.sNP

        # Stagnation and Reset parameters
        self.age = 0
        self.eps = 1e-12  # Tolerance for fitness equality
        self.MyEps = 0.25  # Threshold ratio (25%) for reset
        self.reductions_done = 0

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
        self.CRu_s = 0.8

        # Shared Upper Bound for F
        self.Fu = 1.1

        self.F, self.Cr = None, None

    def initialize(self, args=None, x=None, y=None):
        if x is None:
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                (self.n_individuals, self.ndim_problem),
            )
        else:
            self.n_individuals = x.shape[0]
            self.sNP = min(10, max(1, self.n_individuals // 4))
            self.bNP = self.n_individuals - self.sNP
        if y is None:
            y = np.array([self._evaluate_fitness(xi, args) for xi in x])
        self.F = np.full(self.n_individuals, self.Finit) if self.F is None else self.F
        self.Cr = (
            np.full(self.n_individuals, self.CRinit) if self.Cr is None else self.Cr
        )
        return x, y

    def _reflect_bounds(self, v):
        v = np.where(
            v < self.initial_lower_boundary, 2 * self.initial_lower_boundary - v, v
        )

        v = np.where(
            v > self.initial_upper_boundary, 2 * self.initial_upper_boundary - v, v
        )

        v = np.clip(v, self.initial_lower_boundary, self.initial_upper_boundary)
        return v

    def _check_population_reduction(self, x, y):
        # SYNCHRONIZATION
        actual_size = len(y)
        if actual_size != self.n_individuals:
            self.n_individuals = actual_size
            self.sNP = min(10, max(1, actual_size // 4))
            self.bNP = self.n_individuals - self.sNP

            if len(self.F) != actual_size:
                self.F = np.full(actual_size, self.Finit)
                self.Cr = np.full(actual_size, self.CRinit)

        # REDUCTION LOGIC
        thresholds = [0.25, 0.50, 0.75]
        if self.reductions_done < len(thresholds):
            progress = self.n_function_evaluations / self.max_function_evaluations
            if progress >= thresholds[self.reductions_done]:
                # Calculate the standard halved size for the big population
                new_bNP = self.bNP // 2

                min_allowed_bNP = max(1, self.Nmin - self.sNP)
                new_bNP = max(new_bNP, min_allowed_bNP)

                # Only perform the competition if we are actually shrinking the array
                if new_bNP < self.bNP:
                    part1_idx = np.arange(new_bNP)
                    part2_idx = np.arange(new_bNP, 2 * new_bNP)

                    keep_idx = []
                    for i, j in zip(part1_idx, part2_idx):
                        if j < self.bNP:
                            keep_idx.append(i if y[i] <= y[j] else j)
                        else:
                            keep_idx.append(i)

                    keep_b_idx = np.array(keep_idx, dtype=int)
                    s_idx = np.arange(int(self.bNP), int(self.n_individuals), dtype=int)

                    x = np.concatenate([x[keep_b_idx], x[s_idx]], axis=0)
                    y = np.concatenate([y[keep_b_idx], y[s_idx]], axis=0)
                    self.F = np.concatenate([self.F[keep_b_idx], self.F[s_idx]], axis=0)
                    self.Cr = np.concatenate(
                        [self.Cr[keep_b_idx], self.Cr[s_idx]], axis=0
                    )

                    # Update sizes for the newly reduced population
                    self.bNP = int(len(keep_b_idx))
                    self.n_individuals = int(len(y))

                self.reductions_done += 1

        return x, y

    def _evolve_population(self, x, y, args, is_big=True):
        if self.n_individuals == 0:
            return x, y

        start_idx = 0 if is_big else self.bNP
        end_idx = self.bNP if is_big else self.n_individuals

        f_low = self.Fl_b if is_big else self.Fl_s
        cr_bound = self.CRu_b if is_big else self.CRu_s
        cr_low = self.CRl_b if is_big else self.CRl_s

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

            # Mutation Pool Selection with Extreme RL Fallbacks
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

                # Helper to safely pick a target or fallback sequentially
                def safe_choice(preferred_pool, exclude):
                    valid = [idx for idx in preferred_pool if idx not in exclude]
                    if not valid:
                        valid = [
                            idx
                            for idx in range(self.n_individuals)
                            if idx not in exclude
                        ]
                    return self.rng_optimization.choice(valid) if valid else i

                r1 = safe_choice(range(self.bNP), [i])
                r2 = safe_choice(pool_r2_r3, [i, r1])
                r3 = safe_choice(pool_r2_r3, [i, r1, r2])

            else:
                pool = [idx for idx in range(self.bNP, self.n_individuals) if idx != i]

                # Normal behavior: P_s has enough individuals
                if len(pool) >= 3:
                    r1, r2, r3 = self.rng_optimization.choice(pool, 3, replace=False)
                else:
                    # FALLBACK 1: Try borrowing from the full population without replacement
                    full_pool = [idx for idx in range(self.n_individuals) if idx != i]
                    if len(full_pool) >= 3:
                        r1, r2, r3 = self.rng_optimization.choice(
                            full_pool, 3, replace=False
                        )
                    else:
                        # EXTREME FALLBACK: Population is < 4. We MUST allow replacement.
                        # If population is literally 1, it will just pick `i` three times.
                        full_pool_with_i = list(range(self.n_individuals))
                        r1, r2, r3 = self.rng_optimization.choice(
                            full_pool_with_i, 3, replace=True
                        )

            # Mutation and Reflection
            v = x[r1] + new_F * (x[r2] - x[r3])
            v = self._reflect_bounds(v)

            # Crossover (Rotational Invariant Strategy)
            if new_Cr > 1.0:
                u = v.copy()
            else:
                u = x[i].copy()
                j_rand = self.rng_optimization.integers(0, self.ndim_problem)
                mask = self.rng_optimization.random(self.ndim_problem) <= new_Cr
                mask[j_rand] = True
                u[mask] = v[mask]

            # Evaluate
            new_y = self._evaluate_fitness(u, args)

            # Crowding & Selection
            if is_big:
                # Euclidean distance crowding
                dists = np.sum((x[: self.bNP] - u) ** 2, axis=1)
                target = np.argmin(dists)
            else:
                target = i

            if new_y <= y[target]:
                x[target], y[target] = u, new_y
                self.F[target], self.Cr[target] = new_F, new_Cr

                if is_big and new_y < self.best_so_far_y:
                    self.best_so_far_y = new_y
                    self.age = 0
            elif is_big and target == i:
                self.age += 1

        return x, y

    def iterate(self, x=None, y=None, args=None):
        x, y = self._check_population_reduction(x, y)

        # P_b Reinitialization Check
        if self.bNP > 0:
            best_b_y = np.min(y[: self.bNP])
            eqs_b = np.sum(np.abs(y[: self.bNP] - best_b_y) < self.eps)
            age_limit = 0.1 * self.max_function_evaluations

            if (eqs_b >= self.bNP * self.MyEps) or (self.age >= age_limit):
                x[: self.bNP] = self.rng_initialization.uniform(
                    self.initial_lower_boundary,
                    self.initial_upper_boundary,
                    (self.bNP, self.ndim_problem),
                )
                y[: self.bNP] = np.array(
                    [self._evaluate_fitness(xi, args) for xi in x[: self.bNP]]
                )
                self.F[: self.bNP] = self.Finit
                self.Cr[: self.bNP] = self.CRinit
                self.age = 0

        # P_s Reinitialization Check
        if self.sNP > 0:
            # Safely find the best in the small population
            best_s_idx = self.bNP + np.argmin(y[self.bNP :])
            eqs_s = np.sum(np.abs(y[self.bNP :] - y[best_s_idx]) < self.eps)

            if eqs_s >= self.sNP * self.MyEps:
                best_x_s = x[best_s_idx].copy()
                best_y_s = y[best_s_idx]

                x[self.bNP :] = self.rng_initialization.uniform(
                    self.initial_lower_boundary,
                    self.initial_upper_boundary,
                    (self.sNP, self.ndim_problem),
                )
                y[self.bNP :] = np.array(
                    [self._evaluate_fitness(xi, args) for xi in x[self.bNP :]]
                )
                self.F[self.bNP :] = self.Finit
                self.Cr[self.bNP :] = self.CRinit

                # Elitism: retain the best small-population individual
                x[self.bNP], y[self.bNP] = best_x_s, best_y_s

        # Big Population Generation
        if self.bNP > 0:
            x, y = self._evolve_population(x, y, args, is_big=True)

        # Migration
        # The best individual migrates from P_b to P_s
        if self.bNP > 0 and self.sNP > 0:
            best_overall_idx = np.argmin(y)
            if best_overall_idx < self.bNP:
                worst_s_idx = self.bNP + np.argmax(y[self.bNP :])
                x[worst_s_idx] = x[best_overall_idx].copy()
                y[worst_s_idx] = y[best_overall_idx]
                self.F[worst_s_idx] = self.F[best_overall_idx]
                self.Cr[worst_s_idx] = self.Cr[best_overall_idx]

        # Small Population Generation (repeats m times)
        if self.sNP > 0:
            # m is traditionally bNP // sNP, but must fallback cleanly if bNP is 0
            m = self.bNP // self.sNP if self.bNP > 0 else 1
            m = max(1, m)  # Ensure it executes at least once if P_s is all we have
            for _ in range(m):
                x, y = self._evolve_population(x, y, args, is_big=False)

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
            self.results.update({"x": x, "y": y})
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
            self.Cr = kwargs.get("Cr")
            if self.Cr is not None:
                self.Cr = self.Cr[indices]
            self.F = kwargs.get("F")
            if self.F is not None:
                self.F = self.F[indices]
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
