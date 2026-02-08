import numpy as np

from dynamicalgorithmselection.optimizers.DE.DE import DE


class JDE21(DE):
    def __init__(self, problem, options):
        super().__init__(problem, options)
        self.sNP = 10
        self.bNP = self.n_individuals - self.sNP
        self.age = 0
        self.tao1 = self.tao2 = 0.1
        self.Finit, self.CRinit = 0.5, 0.9
        self.Fu = 1.1
        self.Fl_b, self.CRu_b = 0.1, 1.1
        self.Nmax = self.n_individuals
        self.Nmin = 30

    def initialize(self, args=None, x=None, y=None):
        if x is None:
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                (self.n_individuals, self.ndim_problem),
            )
        if y is None:
            y = np.array([self._evaluate_fitness(xi, args) for xi in x])
        self.F = np.full(self.n_individuals, self.Finit)
        self.Cr = np.full(self.n_individuals, self.CRinit)
        return x, y

    def _mutate_cross_select(self, x, y, indices, args=None):
        NP_sub = len(indices)
        if NP_sub < 4:
            return x, y

        # Self-adaptation
        new_F = np.where(
            self.rng_optimization.random(NP_sub) < self.tao1,
            self.rng_optimization.random(NP_sub) * self.Fu + self.Fl_b,
            self.F[indices],
        )
        new_Cr = np.where(
            self.rng_optimization.random(NP_sub) < self.tao2,
            self.rng_optimization.random(NP_sub) * self.CRu_b,
            self.Cr[indices],
        )

        # Mutation & Crossover
        # Simplified vectorized parent selection
        r1, r2, r3 = [self.rng_optimization.choice(indices, NP_sub) for _ in range(3)]
        vs = x[r1] + new_F[:, np.newaxis] * (x[r2] - x[r3])
        vs = np.clip(vs, self.lower_boundary, self.upper_boundary)

        mask = (
            self.rng_optimization.random((NP_sub, self.ndim_problem))
            < new_Cr[:, np.newaxis]
        )
        us = np.where(mask, vs, x[indices])

        new_y = np.array([self._evaluate_fitness(ui, args) for ui in us])

        # Crowding Selection
        dists = np.linalg.norm(
            x[indices][:, np.newaxis, :] - us[np.newaxis, :, :], axis=2
        )
        closest_sub_idx = np.argmin(dists, axis=0)
        closest_global_idx = indices[closest_sub_idx]

        improved = new_y < y[closest_global_idx]
        for i, idx in enumerate(closest_global_idx):
            if improved[i]:
                x[idx], y[idx] = us[i], new_y[i]
                self.F[idx], self.Cr[idx] = new_F[i], new_Cr[i]
                self.age = 0

        if not np.any(improved):
            self.age += NP_sub
        return x, y

    def iterate(self, x=None, y=None, args=None):
        bNP = x.shape[0] - self.sNP
        # Evolution of big population
        x, y = self._mutate_cross_select(x, y, np.arange(bNP), args)

        # Evolution of small population (repeated)
        small_idx = np.arange(bNP, x.shape[0])
        for _ in range(bNP // self.sNP):
            x, y = self._mutate_cross_select(x, y, small_idx, args)

        progress = self.n_function_evaluations / self.max_function_evaluations
        self.n_individuals = int(round(self.Nmax - progress * (self.Nmax - self.Nmin)))

        self._n_generations += 1
        return x, y

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        x, y = self.initialize(
            args, self.start_conditions.get("x"), self.start_conditions.get("y")
        )
        idx = 0
        while True:
            old_evals = self.n_function_evaluations

            x, y = self.iterate(x, y, args)
            self.results.update(
                {
                    "x": x,
                    "y": y,
                }
            )
            if self._check_terminations():
                break
            idx += 1
            if self.n_function_evaluations == old_evals:
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
