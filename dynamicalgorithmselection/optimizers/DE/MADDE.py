import numpy as np
from dynamicalgorithmselection.optimizers.DE.DE import DE


class MADDE(DE):
    start_condition_parameters = ["x", "y", "archive", "MF", "MCr", "k_idx", "pm"]

    def __init__(self, problem, options):
        super().__init__(problem, options)
        # Constants from MadDE paper/original code
        self.Nmax = self.n_individuals if self.n_individuals else 170
        self.Nmin = options.get("Nmin", 4)
        self.p = 0.18
        self.PqBX = 0.01

        # Adaptive strategy probabilities
        self.pm = np.ones(3) / 3

        # Archive and Memory
        self.NA = int(self.Nmax * 2.1)
        self.archive = np.empty((0, self.ndim_problem))

        # Memory for F and Cr
        self.memory_size = 20  # Standard for SHADE-based
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

    def _choose_F_Cr(self, NP):
        indices = self.rng_optimization.integers(0, self.memory_size, size=NP)
        Cr = self.rng_optimization.normal(loc=self.MCr[indices], scale=0.1, size=NP)
        Cr = np.clip(Cr, 0, 1)

        # Cauchy-like sampling for F
        F = self.MF[indices] + 0.1 * np.tan(
            np.pi * (self.rng_optimization.random(NP) - 0.5)
        )
        while np.any(F <= 0):
            idx = np.where(F <= 0)[0]
            F[idx] = self.MF[indices[idx]] + 0.1 * np.tan(
                np.pi * (self.rng_optimization.random(len(idx)) - 0.5)
            )
        return Cr, np.minimum(1.0, F)

    def _mutate(self, x, y, F, strategy_idx, q, Fa):
        NP = x.shape[0]
        dim = self.ndim_problem
        v = np.zeros_like(x)

        # Indices for 3 strategies
        m0 = strategy_idx == 0
        m1 = strategy_idx == 1
        m2 = strategy_idx == 2

        # p-best and q-best sets
        order = np.argsort(y)
        p_best = x[order[: max(int(self.p * NP), 2)]]
        q_best = x[order[: max(int(q * NP), 2)]]

        # Strategy 0: Current-to-pbest/1 with archive
        if np.any(m0):
            v[m0] = self._ctb_w_arc(x[m0], p_best, self.archive, F[m0])

        # Strategy 1: Current-to-rand/1 with archive
        if np.any(m1):
            v[m1] = self._ctr_w_arc(x[m1], self.archive, F[m1])

        # Strategy 2: Weighted Rand-to-best
        if np.any(m2):
            v[m2] = self._weighted_rtb(x[m2], q_best, F[m2], Fa)

        return v

    def iterate(self, x=None, y=None, args=None):
        if x is None or y is None:
            raise ValueError("x and y must be provided for iteration.")
        NP = x.shape[0]
        dim = self.ndim_problem
        FEs, MaxFEs = self.n_function_evaluations, self.max_function_evaluations

        # Linear parameters for MadDE
        q = 2 * self.p - self.p * FEs / MaxFEs
        Fa = 0.5 + 0.5 * FEs / MaxFEs

        # 1. Parameter sampling
        Cr, F = self._choose_F_Cr(NP)
        mu = self.rng_optimization.choice(3, size=NP, p=self.pm)

        # 2. Mutation
        v = self._mutate(x, y, F, mu, q, Fa)

        # Boundary handling (MadDE specific)
        low, high = self.lower_boundary, self.upper_boundary
        v = np.where(v < low, (x + low) / 2, v)
        v = np.where(v > high, (x + high) / 2, v)

        # 3. Crossover (Binomial + qBX)
        u = np.zeros_like(x)
        rvs = self.rng_optimization.random(NP)

        # Standard Binomial
        bu_idx = rvs > self.PqBX
        if np.any(bu_idx):
            u[bu_idx] = self._binomial(x[bu_idx], v[bu_idx], Cr[bu_idx])

        # quasi-Best Crossover (qBX)
        qu_idx = rvs <= self.PqBX
        if np.any(qu_idx):
            # Pick qbest from combined population and archive
            combined = np.vstack([x, self.archive]) if len(self.archive) > 0 else x
            q_limit = max(int(q * len(combined)), 2)
            q_best_combined = combined[
                np.argsort(np.concatenate([y, np.full(len(self.archive), np.inf)]))[
                    :q_limit
                ]
            ]
            cross_qbest = q_best_combined[
                self.rng_optimization.integers(0, len(q_best_combined), np.sum(qu_idx))
            ]
            u[qu_idx] = self._binomial(cross_qbest, v[qu_idx], Cr[qu_idx])

        # 4. Evaluation and Selection
        new_y = np.array([self._evaluate_fitness(ui, args) for ui in u])
        optim = new_y < y

        if np.any(optim):
            # Archive update
            self.archive = np.vstack([self.archive, x[optim]])
            if len(self.archive) > self.NA:
                self.archive = self.archive[
                    self.rng_optimization.choice(
                        len(self.archive), self.NA, replace=False
                    )
                ]

            # Memory and Strategy probability update
            df = np.maximum(0, y - new_y)
            self._update_memory(F[optim], Cr[optim], df[optim])
            self._update_pm(df, mu)

            x[optim], y[optim] = u[optim], new_y[optim]

        # 5. NLPSR
        x, y = self._nlpsr(x, y)

        self._n_generations += 1
        return x, y

    def _update_pm(self, df, mu):
        count_S = np.zeros(3)
        for i in range(3):
            if np.any(mu == i):
                count_S[i] = np.mean(df[mu == i])

        if np.sum(count_S) > 0:
            self.pm = np.maximum(
                0.1, np.minimum(0.9, count_S / (np.sum(count_S) + 1e-15))
            )
            self.pm /= np.sum(self.pm)
        else:
            self.pm = np.ones(3) / 3

    def _nlpsr(self, x, y):
        FEs, MaxFEs = self.n_function_evaluations, self.max_function_evaluations
        new_NP = int(
            np.round(
                self.Nmax
                + (self.Nmin - self.Nmax) * np.power(FEs / MaxFEs, 1 - FEs / MaxFEs)
            )
        )
        if new_NP < x.shape[0]:
            idx = np.argsort(y)[:new_NP]
            x, y = x[idx], y[idx]
            self.n_individuals = new_NP
            self.NA = int(max(new_NP * 2.1, self.Nmin))
        return x, y

    # Helper mutation methods (Vectorized)
    def _ctb_w_arc(self, x, best, archive, F):
        NP = x.shape[0]
        xb = best[self.rng_optimization.integers(0, len(best), NP)]
        r1 = self.rng_optimization.integers(0, NP, NP)
        combined = np.vstack([x, archive]) if len(archive) > 0 else x
        r2 = self.rng_optimization.integers(0, len(combined), NP)
        return (
            x + F[:, np.newaxis] * (xb - x) + F[:, np.newaxis] * (x[r1] - combined[r2])
        )

    def _ctr_w_arc(self, x, archive, F):
        NP = x.shape[0]
        r1 = self.rng_optimization.integers(0, NP, NP)
        combined = np.vstack([x, archive]) if len(archive) > 0 else x
        r2 = self.rng_optimization.integers(0, len(combined), NP)
        return x + F[:, np.newaxis] * (x[r1] - combined[r2])

    def _weighted_rtb(self, x, best, F, Fa):
        NP = x.shape[0]
        xb = best[self.rng_optimization.integers(0, len(best), NP)]
        r1 = self.rng_optimization.integers(0, NP, NP)
        r2 = self.rng_optimization.integers(0, NP, NP)
        return F[:, np.newaxis] * x[r1] + (F * Fa)[:, np.newaxis] * (xb - x[r2])

    def _binomial(self, x, v, Cr):
        NP, dim = x.shape
        jrand = self.rng_optimization.integers(dim, size=NP)
        mask = self.rng_optimization.random((NP, dim)) < Cr[:, np.newaxis]
        u = np.where(mask, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def _update_memory(self, SF, SCr, df):
        if len(SF) > 0:
            w = df / (np.sum(df) + 1e-15)
            self.MF[self.k_idx] = np.sum(w * (SF**2)) / (np.sum(w * SF) + 1e-15)
            self.MCr[self.k_idx] = np.sum(w * SCr)
            self.k_idx = (self.k_idx + 1) % self.memory_size

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        x, y = self.initialize(
            args, self.start_conditions.get("x"), self.start_conditions.get("y")
        )
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
