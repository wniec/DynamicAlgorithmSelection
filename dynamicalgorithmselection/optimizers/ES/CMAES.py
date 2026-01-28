import numpy as np  # engine for numerical computing

from dynamicalgorithmselection.optimizers.ES.ES import ES


class CMAES(ES):
    start_condition_parameters = [
        "mean",
        "x",
        "p_c",
        "p_s",
        "cm",
        "e_ve",
        "e_va",
        "d",
        "y",
    ]

    def __init__(self, problem, options):
        self.options = options
        ES.__init__(self, problem, options | {"sigma": 1.5})
        assert self.n_individuals >= 2
        self._w, self._mu_eff, self._mu_eff_minus = (
            None,
            None,
            None,
        )  # variance effective selection mass
        # c_s (c_σ) -> decay rate for the cumulating path for the step-size control
        self.c_s, self.d_sigma = (
            None,
            None,
        )  # for cumulative step-length adaptation (CSA)
        self._p_s_1, self._p_s_2 = None, None  # for evolution path update of CSA
        self._p_c_1, self._p_c_2 = None, None  # for evolution path update of CMA
        # c_c -> decay rate for cumulating path for the rank-one update of CMA
        # c_1 -> learning rate for the rank-one update of CMA
        # c_w (c_μ) -> learning rate for the rank-µ update of CMA
        self.c_c, self.c_1, self.c_w, self._alpha_cov = (
            None,
            None,
            None,
            2.0,
        )  # for CMA (c_w -> c_μ)
        self._save_eig = options.get(
            "_save_eig", False
        )  # whether or not save eigenvalues and eigenvectors

    def _set_c_c(self):
        """Set decay rate of evolution path for the rank-one update of CMA."""
        return (4.0 + self._mu_eff / self.ndim_problem) / (
            self.ndim_problem + 4.0 + 2.0 * self._mu_eff / self.ndim_problem
        )

    def _set_c_w(self):
        return np.minimum(
            1.0 - self.c_1,
            self._alpha_cov
            * (1.0 / 4.0 + self._mu_eff + 1.0 / self._mu_eff - 2.0)
            / (
                np.square(self.ndim_problem + 2.0)
                + self._alpha_cov * self._mu_eff / 2.0
            ),
        )

    def _set_d_sigma(self):
        return (
            1.0
            + 2.0
            * np.maximum(
                0.0, np.sqrt((self._mu_eff - 1.0) / (self.ndim_problem + 1.0)) - 1.0
            )
            + self.c_s
        )

    def initialize(
        self,
        is_restart=False,
        mean=None,
        x=None,
        p_c=None,
        p_s=None,
        cm=None,
        e_ve=None,
        e_va=None,
        d=None,
        y=None,
    ):
        w_a = np.log((self.n_individuals + 1.0) / 2.0) - np.log(
            np.arange(self.n_individuals) + 1.0
        )  # w_apostrophe
        self._mu_eff = np.square(np.sum(w_a[: self.n_parents])) / np.sum(
            np.square(w_a[: self.n_parents])
        )
        self._mu_eff_minus = np.square(np.sum(w_a[self.n_parents :])) / np.sum(
            np.square(w_a[self.n_parents :])
        )
        self.c_s = self.options.get(
            "c_s", (self._mu_eff + 2.0) / (self.ndim_problem + self._mu_eff + 5.0)
        )
        self.d_sigma = self.options.get("d_sigma", self._set_d_sigma())
        self.c_c = self.options.get("c_c", self._set_c_c())
        self.c_1 = self.options.get(
            "c_1", self._alpha_cov / (np.square(self.ndim_problem + 1.3) + self._mu_eff)
        )
        self.c_w = self.options.get("c_w", self._set_c_w())
        w_min = np.min(
            [
                1.0 + self.c_1 / self.c_w,
                1.0 + 2.0 * self._mu_eff_minus / (self._mu_eff + 2.0),
                (1.0 - self.c_1 - self.c_w) / (self.ndim_problem * self.c_w),
            ]
        )
        self._w = np.where(
            w_a >= 0,
            1.0 / np.sum(w_a[w_a > 0]) * w_a,
            w_min / (-np.sum(w_a[w_a < 0])) * w_a,
        )
        self._p_s_1, self._p_s_2 = (
            1.0 - self.c_s,
            np.sqrt(self.c_s * (2.0 - self.c_s) * self._mu_eff),
        )
        self._p_c_1, self._p_c_2 = (
            1.0 - self.c_c,
            np.sqrt(self.c_c * (2.0 - self.c_c) * self._mu_eff),
        )
        x = (
            x if x is not None else np.zeros((self.n_individuals, self.ndim_problem))
        )  # a population of search points (individuals, offspring)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        p_s = (
            p_s if p_s is not None else np.zeros((self.ndim_problem,))
        )  # evolution path (p_σ) for cumulative step-length adaptation (CSA)
        p_c = (
            p_c if p_c is not None else np.zeros((self.ndim_problem,))
        )  # evolution path for covariance matrix adaptation (CMA)
        cm = (
            cm if cm is not None else np.eye(self.ndim_problem)
        )  # covariance matrix of Gaussian search distribution
        e_ve = (
            e_ve if e_ve is not None else np.eye(self.ndim_problem)
        )  # eigenvectors of `cm` (orthogonal matrix)
        e_va = (
            e_va if e_va is not None else np.ones((self.ndim_problem,))
        )  # square roots of eigenvalues of `cm` (in diagonal rather matrix form)
        y = (
            y if y is not None else np.zeros((self.n_individuals,))
        )  # fitness (no evaluation)
        d = d if d is not None else np.zeros((self.n_individuals, self.ndim_problem))
        self._list_initial_mean.append(np.copy(mean))
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def iterate(
        self, x=None, mean=None, e_ve=None, e_va=None, y=None, d=None, args=None
    ):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return x, y, d
            # produce a spherical (isotropic) Gaussian distribution (Nikolaus Hansen, 2023)
            z = self.rng_optimization.standard_normal(
                (self.ndim_problem,)
            )  # Gaussian noise for mutation
            d[k] = np.dot(e_ve @ np.diag(e_va), z)
            x[k] = mean + self.sigma * d[k]  # offspring individual
            y[k] = self._evaluate_fitness(x[k], args, d=d[k], e_ve=e_ve, e_va=e_va)
        return x, y, d

    def update_distribution(
        self, x=None, p_s=None, p_c=None, cm=None, e_ve=None, e_va=None, y=None, d=None
    ):
        order = np.argsort(y)  # to rank all offspring individuals
        wd = np.dot(self._w[: self.n_parents], d[order[: self.n_parents]])
        # update distribution mean via weighted recombination
        mean = np.dot(self._w[: self.n_parents], x[order[: self.n_parents]])
        # update global step-size: cumulative path length control / cumulative step-size control /
        #   cumulative step length adaptation (CSA)
        cm_minus_half = e_ve @ np.diag(1.0 / e_va) @ e_ve.T
        p_s = self._p_s_1 * p_s + self._p_s_2 * np.dot(cm_minus_half, wd)
        self.sigma *= np.exp(
            self.c_s / self.d_sigma * (np.linalg.norm(p_s) / self._e_chi - 1.0)
        )
        # update covariance matrix (CMA)
        h_s = (
            1.0
            if np.linalg.norm(p_s)
            / np.sqrt(1.0 - np.power(1.0 - self.c_s, 2 * (self._n_generations + 1)))
            < (1.4 + 2.0 / (self.ndim_problem + 1.0)) * self._e_chi
            else 0.0
        )
        p_c = self._p_c_1 * p_c + h_s * self._p_c_2 * wd
        w_o = self._w * np.where(
            self._w >= 0,
            1.0,
            self.ndim_problem
            / (np.square(np.linalg.norm(cm_minus_half @ d.T, axis=0)) + 1e-8),
        )
        cm = (
            1.0
            + self.c_1 * (1.0 - h_s) * self.c_c * (2.0 - self.c_c)
            - self.c_1
            - self.c_w * np.sum(self._w)
        ) * cm + self.c_1 * np.outer(p_c, p_c)  # rank-one update
        for i in range(
            self.n_individuals
        ):  # rank-μ update (to estimate variances of sampled *steps*)
            cm += self.c_w * w_o[i] * np.outer(d[order[i]], d[order[i]])
        # do eigen-decomposition and return both eigenvalues and eigenvectors
        cm = (cm + np.transpose(cm)) / 2.0  # to ensure symmetry of covariance matrix
        # return eigenvalues and eigenvectors of a symmetric matrix
        e_va, e_ve = np.linalg.eigh(cm)  # e_va -> eigenvalues, e_ve -> eigenvectors
        e_va = np.sqrt(
            np.where(e_va < 0.0, 1e-8, e_va)
        )  # to avoid negative eigenvalues
        # e_va: squared root of eigenvalues -> interpreted as individual step-sizes and its diagonal entries are
        #       standard deviations of different components (from Nikolaus Hansen, 2023)
        cm = (
            e_ve @ np.diag(np.square(e_va)) @ np.transpose(e_ve)
        )  # to recover covariance matrix
        return mean, p_s, p_c, cm, e_ve, e_va

    def restart_reinitialize(
        self,
        x=None,
        mean=None,
        p_s=None,
        p_c=None,
        cm=None,
        e_ve=None,
        e_va=None,
        y=None,
        d=None,
    ):
        if ES.restart_reinitialize(self, y):
            x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize(True)
        return x, mean, p_s, p_c, cm, e_ve, e_va, y, d

    def optimize(
        self, fitness_function=None, args=None
    ):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)

        self.sigma = self.start_conditions.get("sigma", self.sigma)
        mean = self.start_conditions.get("mean", None)
        x = self.start_conditions.get("x", None)
        p_c = self.start_conditions.get("p_c", None)
        p_s = self.start_conditions.get("p_s", None)
        cm = self.start_conditions.get("cm", None)
        e_ve = self.start_conditions.get("e_ve", None)
        e_va = self.start_conditions.get("e_va", None)
        d = self.start_conditions.get("d", None)
        y = self.start_conditions.get("y", None)

        x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.initialize(
            is_restart=False,
            mean=mean,
            x=x,
            p_c=p_c,
            p_s=p_s,
            cm=cm,
            e_ve=e_ve,
            e_va=e_va,
            d=d,
            y=y,
        )
        while True:
            # sample and evaluate offspring population
            x, y, d = self.iterate(x, mean, e_ve, e_va, y, d, args)
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, p_s, p_c, cm, e_ve, e_va = self.update_distribution(
                x, p_s, p_c, cm, e_ve, e_va, y, d
            )
            self.results.update(
                {
                    "p_c": p_c,
                    "p_s": p_s,
                    "cm": cm,
                    "e_va": e_va,
                    "e_ve": e_ve,
                    "d": d,
                    "x": x,
                    "y": y,
                    "mean": mean,
                }
            )
            if self.is_restart:
                x, mean, p_s, p_c, cm, e_ve, e_va, y, d = self.restart_reinitialize(
                    x, mean, p_s, p_c, cm, e_ve, e_va, y, d
                )
                self.results.update(
                    {
                        "p_c": p_c,
                        "p_s": p_s,
                        "cm": cm,
                        "e_va": e_va,
                        "e_ve": e_ve,
                        "d": d,
                        "x": x,
                        "y": y,
                        "mean": mean,
                    }
                )
        self.results.update(
            {
                "p_c": p_c,
                "p_s": p_s,
                "cm": cm,
                "e_va": e_va,
                "e_ve": e_ve,
                "d": d,
                "x": x,
                "y": y,
                "mean": mean,
            }
        )
        self.results.update(
            {
                "p_c": p_c,
                "p_s": p_s,
                "cm": cm,
                "e_va": e_va,
                "e_ve": e_ve,
                "d": d,
                "x": x,
                "y": y,
                "mean": mean,
            }
        )
        results = self._collect(fitness, y, mean)
        # by default do *NOT* save eigenvalues and eigenvectors (with *quadratic* space complexity)
        if self._save_eig:
            results["e_va"], results["e_ve"] = e_va, e_ve
        return results

    def set_data(
        self,
        x=None,
        y=None,
        mean=None,
        p_c=None,
        p_s=None,
        cm=None,
        d=None,
        e_ve=None,
        e_va=None,
        *args,
        **kwargs,
    ):
        if x is None or y is None:
            self.start_conditions = {"x": None, "y": None, "mean": None}
        elif not isinstance(y, np.ndarray):
            loc = locals()
            self.start_conditions = {
                i: loc.get(i, None)
                for i in [
                    "yp_c",
                    "p_s",
                    "cm",
                    "e_ve",
                    "e_va",
                    "d",
                ]
            }
        else:
            indices = np.argsort(y)[: self.n_individuals]
            loc = locals()
            start_conditions = {
                i: loc.get(i, None)
                for i in [
                    "yp_c",
                    "p_s",
                    "cm",
                    "e_ve",
                    "e_va",
                    "d",
                ]
            }
            mean = x[indices].mean(axis=0)
            stds = np.std(x[indices], axis=0)
            sigma = np.max(stds)
            sigma = max(sigma, 1e-8)
            start_conditions.update(
                {"x": x[indices], "y": y[indices], "mean": mean, "sigma": sigma}
            )
            self.start_conditions = start_conditions
        if self.start_conditions.get("d") is not None:
            self.start_conditions["d"] = self.start_conditions["d"][
                : self.n_individuals
            ]
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
