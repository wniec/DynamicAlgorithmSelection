import numpy as np
from dynamicalgorithmselection.optimizers.ES.ES import ES


class LMCMAES(ES):
    def __init__(self, problem, options):
        ES.__init__(self, problem, options | {"sigma": 1.5})
        self.m = options.get(
            "m", 4 + int(3 * np.log(self.ndim_problem))
        )  # number of direction vectors
        self.n_steps = options.get(
            "n_steps", self.m
        )  # target number of generations between vectors
        self.c_c = options.get(
            "c_c", 1.0 / self.m
        )  # learning rate for evolution path update
        self.c_1 = options.get("c_1", 1.0 / (10.0 * np.log(self.ndim_problem + 1.0)))
        self.c_s = options.get(
            "c_s", 0.3
        )  # learning rate for population success rule (PSR)
        self.d_s = options.get("d_s", 1.0)  # damping parameter for PSR
        self.z_star = options.get("z_star", 0.25)  # target success rate for PSR
        self._a = np.sqrt(1.0 - self.c_1)
        self._c = 1.0 / np.sqrt(1.0 - self.c_1)
        self._bd_1 = np.sqrt(1.0 - self.c_1)
        self._bd_2 = self.c_1 / (1.0 - self.c_1)
        self._p_c_1 = 1.0 - self.c_c
        self._p_c_2 = None
        self._j = None
        self._l = None
        self._it = None
        self._rr = None  # for PSR

    def _initialize_mean(self, is_restart=False):
        if is_restart or (self.mean is None):
            mean = self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary
            )
        else:
            mean = np.copy(self.mean)
        self.mean = np.copy(mean)
        return mean

    def initialize(
        self,
        is_restart=False,
        mean=None,
        x=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        y=None,
    ):
        mean = mean if mean is not None else self._initialize_mean(is_restart)
        x = x if x is not None else np.zeros((self.n_individuals, self.ndim_problem))
        p_c = p_c if p_c is not None else np.zeros((self.ndim_problem,))
        s = s if s is not None else 0.0
        vm = vm if vm is not None else np.zeros((self.m, self.ndim_problem))
        pm = pm if pm is not None else np.zeros((self.m, self.ndim_problem))
        b = b if b is not None else np.zeros((self.m,))
        d = d if d is not None else np.zeros((self.m,))
        y = y if y is not None else np.zeros((self.n_individuals,))

        self._p_c_2 = np.sqrt(self.c_c * (2.0 - self.c_c) * self._mu_eff)
        self._rr = np.arange(self.n_individuals * 2, 0, -1) - 1
        self._j = [None] * self.m
        self._l = [None] * self.m
        self._it = 0
        return mean, x, p_c, s, vm, pm, b, d, y

    def _a_z(self, z=None, pm=None, vm=None, b=None):  # Algorithm 3 Az()
        x = np.copy(z)
        for t in range(self._it):
            x = self._a * x + b[self._j[t]] * np.dot(vm[self._j[t]], z) * pm[self._j[t]]
        return x

    def iterate(self, mean=None, x=None, pm=None, vm=None, y=None, b=None, args=None):
        sign, a_z = 1, np.empty((self.ndim_problem,))  # for mirrored sampling
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            if sign == 1:
                z = self.rng_optimization.standard_normal((self.ndim_problem,))
                a_z = self._a_z(z, pm, vm, b)

            mutation_step = sign * self.sigma * a_z
            if np.any(np.isnan(mutation_step)) or np.any(np.isinf(mutation_step)):
                # Fallback to prevent crash, effectively skipping this mutation
                mutation_step = np.zeros_like(mutation_step)

            x[k] = mean + mutation_step
            y[k] = self._evaluate_fitness(x[k], args)
            sign *= -1
        return x, y

    def _a_inv_z(self, v=None, vm=None, d=None, i=None):  # Algorithm 4 Ainvz()
        x = np.copy(v)
        for t in range(0, i):
            dot_prod = np.dot(vm[self._j[t]], x)
            x = self._c * x - d[self._j[t]] * dot_prod * vm[self._j[t]]
        return x

    def _update_distribution(
        self,
        mean=None,
        x=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        y=None,
        y_bak=None,
    ):
        mean_bak = np.dot(self._w, x[np.argsort(y)[: self.n_parents]])

        safe_sigma = np.clip(self.sigma, 1e-20, 1e20)

        p_c = self._p_c_1 * p_c + self._p_c_2 * (mean_bak - mean) / safe_sigma

        i_min = 1
        if self._n_generations < self.m:
            self._j[self._n_generations] = self._n_generations
        else:
            d_min = self._l[self._j[i_min]] - self._l[self._j[i_min - 1]]
            for j in range(2, self.m):
                d_cur = self._l[self._j[j]] - self._l[self._j[j - 1]]
                if d_cur < d_min:
                    d_min, i_min = d_cur, j
            i_min = 0 if d_min >= self.n_steps else i_min
            updated = self._j[i_min]
            for j in range(i_min, self.m - 1):
                self._j[j] = self._j[j + 1]
            self._j[self.m - 1] = updated

        self._it = np.minimum(self._n_generations + 1, self.m)
        self._l[self._j[self._it - 1]] = self._n_generations
        pm[self._j[self._it - 1]] = p_c

        for i in range(0 if i_min == 1 else i_min, self._it):
            vm[self._j[i]] = self._a_inv_z(pm[self._j[i]], vm, d, i)
            v_n = np.dot(vm[self._j[i]], vm[self._j[i]])

            # If v_n is 0 or NaN, b and d will explode.
            if v_n < 1e-20:
                v_n = 1e-20

            bd_3 = np.sqrt(1.0 + self._bd_2 * v_n)
            b[self._j[i]] = self._bd_1 / v_n * (bd_3 - 1.0)
            d[self._j[i]] = 1.0 / (self._bd_1 * v_n) * (1.0 - 1.0 / bd_3)

        if self._n_generations > 0:
            r = np.argsort(np.hstack((y, y_bak)))
            z_psr = np.sum(
                self._rr[r < self.n_individuals] - self._rr[r >= self.n_individuals]
            )
            z_psr = z_psr / np.power(self.n_individuals, 2) - self.z_star
            s = (1.0 - self.c_s) * s + self.c_s * z_psr

            # FIX 5: Clamp s to prevent sigma explosion via exp()
            # exp(100) is huge, exp(709) is infinity. Clamp s to safe range.
            s = np.clip(s, -50.0, 50.0)

            self.sigma *= np.exp(s / self.d_s)

        return mean_bak, p_c, s, vm, pm, b, d

    def restart_reinitialize(
        self,
        mean=None,
        x=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        y=None,
    ):
        if self.is_restart and ES.restart_reinitialize(self, y):
            mean, x, p_c, s, vm, pm, b, d, y = self.initialize(True)
            self.d_s *= 2.0
        return mean, x, p_c, s, vm, pm, b, d, y

    def optimize(self, fitness_function=None, args=None):
        fitness = ES.optimize(self, fitness_function)
        self.sigma = self.start_conditions.get("sigma", self.sigma)
        mean = self.start_conditions.get("mean", None)
        x = self.start_conditions.get("x", None)
        p_c = self.start_conditions.get("p_c", None)
        s = self.start_conditions.get("s", None)
        vm = self.start_conditions.get("vm", None)
        pm = self.start_conditions.get("pm", None)
        b = self.start_conditions.get("b", None)
        d = self.start_conditions.get("d", None)
        y = self.start_conditions.get("y", None)
        mean, x, p_c, s, vm, pm, b, d, y = self.initialize(
            args, mean, x, p_c, s, vm, pm, b, d, y
        )
        while not self.termination_signal:
            y_bak = np.copy(y)
            x, y = self.iterate(mean, x, pm, vm, y, b, args)

            mean, p_c, s, vm, pm, b, d = self._update_distribution(
                mean, x, p_c, s, vm, pm, b, d, y, y_bak
            )
            self.results.update(
                {
                    "p_c": p_c,
                    "s": s,
                    "vm": vm,
                    "pm": pm,
                    "b": b,
                    "d": d,
                    "x": x,
                    "y": y,
                    "mean": mean,
                }
            )
            if self._check_terminations():
                break
            self._print_verbose_info(fitness, y)
            self._n_generations += 1
            mean, x, p_c, s, vm, pm, b, d, y = self.restart_reinitialize(
                mean, x, p_c, s, vm, pm, b, d, y
            )
        self.results.update(
            {
                "p_c": p_c,
                "s": s,
                "vm": vm,
                "pm": pm,
                "b": b,
                "d": d,
                "x": x,
                "y": y,
                "mean": mean,
            }
        )
        results = self._collect(fitness, y, mean)
        return results

    def set_data(
        self,
        x=None,
        y=None,
        mean=None,
        p_c=None,
        s=None,
        vm=None,
        pm=None,
        b=None,
        d=None,
        *args,
        **kwargs,
    ):
        if x is None or y is None:
            self.start_conditions = {"x": None, "y": None, "mean": None}
        elif not isinstance(y, np.ndarray):
            loc = locals()
            self.start_conditions = {
                i: loc.get(i, None) for i in ("cf", "best_so_far_y", "p_s", "p_c")
            }
        else:
            indices = np.argsort(y)[: self.n_individuals]
            loc = locals()
            start_conditions = {
                i: loc.get(i, None) for i in ("p_c", "s", "vm", "pm", "b", "d")
            }
            mean = x[indices].mean(axis=0)
            stds = np.std(x[indices], axis=0)
            sigma = np.max(stds)
            sigma = max(sigma, 1e-8)
            start_conditions.update(
                {"x": x[indices], "y": y[indices], "mean": mean, "sigma": sigma}
            )
            self.start_conditions = start_conditions
        self.best_so_far_x = kwargs.get("best_x", None)
        self.best_so_far_y = kwargs.get("best_y", float("inf"))
