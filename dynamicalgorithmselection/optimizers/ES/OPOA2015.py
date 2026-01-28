from typing import Optional

import numpy as np
import numba as nb

from dynamicalgorithmselection.optimizers.ES.ES import ES


@nb.jit(nopython=True)
def cholesky_update(rm, z, downdate):
    rm, z, alpha, beta = rm.T, z, np.empty_like(z), np.empty_like(z)
    alpha[-1], beta[-1] = 1.0, 1.0
    sign = -1 if downdate else 1
    for r in range(len(z)):
        a = z[r] / rm[r, r]
        alpha[r] = alpha[r - 1] + sign * np.power(a, 2)
        beta[r] = np.sqrt(alpha[r])
        z[r + 1 :] -= a * rm[r, r + 1 :]
        rm[r, r:] *= beta[r] / beta[r - 1]
        rm[r, r + 1 :] += sign * a / (beta[r] * beta[r - 1]) * z[r + 1 :]
    return rm.T


class OPOA2015(ES):
    def __init__(self, problem, options):
        self.mean_history = []
        self.y_history = []
        options["n_individuals"] = 1  # mandatory setting
        options["n_parents"] = 1  # mandatory setting
        ES.__init__(self, problem, options | {"sigma": 0.9})
        if self.lr_sigma is None:
            self.lr_sigma = 1.0 / (1.0 + self.ndim_problem / 2.0)
        self.p_ts = options.get("p_ts", 2.0 / 11.0)
        self.c_p = options.get("c_p", 1.0 / 12.0)
        self.c_c = options.get("c_c", 2.0 / (self.ndim_problem + 2.0))
        self.c_cov = options.get("c_cov", 2.0 / (np.power(self.ndim_problem, 2) + 6.0))
        self.p_t = options.get("p_t", 0.44)
        self.c_m = options.get("c_m", 0.4 / (np.power(self.ndim_problem, 1.6) + 1.0))
        self.k = options.get("k", 5)
        self._ancestors = []
        self._c_cf = 1.0 - self.c_cov + self.c_cov * self.c_c * (2.0 - self.c_c)

    def initialize(
        self,
        mean=None,
        y=None,
        cf=None,
        best_so_far_y=None,
        p_s=None,
        p_c=None,
        args=None,
        is_restart=False,
    ):
        mean = self._initialize_mean(is_restart) if mean is None else mean
        y = (
            self._evaluate_fitness(
                x=mean,
                args=args,
                cf=cf,
                p_s=p_s,
                p_c=p_c,
            )
            if y is None
            else y
        )
        cf = (
            np.diag(
                np.ones(
                    self.ndim_problem,
                )
            )
            if cf is None
            else cf
        )
        best_so_far_y = float(np.copy(y)) if best_so_far_y is None else best_so_far_y
        p_s = self.p_ts if p_s is None else p_s
        p_c = np.zeros((self.ndim_problem,)) if p_c is None else p_c
        return mean, y, cf, best_so_far_y, p_s, p_c

    def _cholesky_update(
        self, cf=None, alpha=None, beta=None, v=None
    ):  # triangular rank-one update
        assert self.ndim_problem == v.size
        if beta < 0:
            downdate, beta = True, -beta
        else:
            downdate = False
        return cholesky_update(
            np.sqrt(max(alpha, 1e-8)) * cf, np.sqrt(beta + 1e-8) * v, downdate
        )

    def iterate(
        self, mean=None, cf=None, best_so_far_y=None, p_s=None, p_c=None, args=None
    ):
        # sample and evaluate only one offspring
        z = self.rng_optimization.standard_normal((self.ndim_problem,))
        cf_z = np.dot(cf, z)
        x = mean + self.sigma * cf_z
        y = (
            self._evaluate_fitness(
                x=x,
                args=args,
                cf=cf,
                p_s=p_s,
                p_c=p_c,
            )
        )
        if y <= best_so_far_y:
            self._ancestors.append(y)
            mean, best_so_far_y = x, y
            p_s = (1.0 - self.c_p) * p_s + self.c_p
            is_better = True
        else:
            p_s *= 1.0 - self.c_p
            is_better = False
        self.sigma *= np.exp(self.lr_sigma * (p_s - self.p_ts) / (1.0 - self.p_ts))
        if p_s >= self.p_t:
            p_c *= 1.0 - self.c_c
            cf = self._cholesky_update(cf, self._c_cf, self.c_cov, p_c)
        elif is_better:
            p_c = (1.0 - self.c_c) * p_c + np.sqrt(self.c_c * (2.0 - self.c_c)) * cf_z
            cf = self._cholesky_update(cf, 1.0 - self.c_cov, self.c_cov, p_c)
        elif len(self._ancestors) >= self.k and y > self._ancestors[-self.k]:
            del self._ancestors[0]
            c_m = np.minimum(self.c_m, 1.0 / (2.0 * np.dot(z, z) - 1.0))
            cf = self._cholesky_update(cf, 1.0 + c_m, -c_m, cf_z)
        self._n_generations += 1
        self.results.update(
            {
                "cf": cf,
                "best_so_far_y": best_so_far_y,
                "p_s": p_s,
                "p_c": p_c,
            }
        )
        self.mean_history.append(mean)
        self.y_history.append(y)
        return mean, y, cf, best_so_far_y, p_s, p_c

    def restart_reinitialize(
        self,
        mean=None,
        y=None,
        cf=None,
        best_so_far_y=None,
        p_s=None,
        p_c=None,
        fitness=None,
        args=None,
    ):
        self._list_fitness.append(best_so_far_y)
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (
                self._list_fitness[-self.stagnation] - self._list_fitness[-1]
            ) < self.fitness_diff
        is_restart = bool(is_restart_1) or bool(is_restart_2)
        if self.is_restart and is_restart:
            self._print_verbose_info(fitness, y, True)
            if self.verbose:
                print(" ....... *** restart *** .......")
            self._n_restart += 1
            self._list_generations.append(self._n_generations)  # for each restart
            self._n_generations = 0
            self.sigma = np.copy(self._sigma_bak)
            mean, y, cf, best_so_far_y, p_s, p_c = self.initialize(args, True)
            self._list_fitness = [best_so_far_y]
            self._ancestors = []
        return mean, y, cf, best_so_far_y, p_s, p_c

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)

        mean = self.start_conditions.get("mean", None)
        p_c = self.start_conditions.get("p_c", None)
        p_s = self.start_conditions.get("p_s", None)
        best_so_far_y = self.start_conditions.get("best_so_far_y", None)
        cf = self.start_conditions.get("cf", None)
        y = self.start_conditions.get("y", None)

        mean, y, cf, best_so_far_y, p_s, p_c = self.initialize(
            mean, y, cf, best_so_far_y, p_s, p_c, args
        )
        while not self._check_terminations():
            self._print_verbose_info(fitness, y)
            mean, y, cf, best_so_far_y, p_s, p_c = self.iterate(
                mean, cf, best_so_far_y, p_s, p_c, args
            )
            mean, y, cf, best_so_far_y, p_s, p_c = self.restart_reinitialize(
                mean, y, cf, best_so_far_y, p_s, p_c, fitness, args
            )
        return self._collect(fitness, y, mean)

    def set_data(
        self,
        mean=None,
        y=None,
        cf=None,
        best_so_far_y=None,
        p_s=None,
        p_c=None,
        x=None,
        *args,
        **kwargs,
    ):
        mean = (
            mean
            if mean is not None
            else (np.mean(x, axis=0) if x is not None else None)
        )
        y = y if isinstance(y, float) else None
        self.start_conditions = {
            "mean": mean,
            "y": y,
            "cf": cf,
            "best_so_far_y": best_so_far_y,
            "p_s": p_s,
            "p_c": p_c,
        }

        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))

    def get_data(self, n_individuals: Optional[int] = None):
        pop_data = ["x", "y"]
        best_indices = sorted(
            [i for i in range(len(self.y_history))],
            key=lambda x: self.y_history[x],
        )[:n_individuals]
        x = np.array(self.mean_history)[best_indices]
        y = np.array(self.y_history)[best_indices]
        return self.results | {"x": x, "y": y} or self.start_conditions
