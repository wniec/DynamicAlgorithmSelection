from typing import Optional

import numpy as np
from dynamicalgorithmselection.optimizers.DS.DS import DS


def _minimize_scalar_bounded(
    func,
    bounds,
    max_function_evaluations,
    fitness_threshold,
    tol=1e-5,
    max_iterations=500,
):
    # this is adopted from https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py
    #   with slight modifications
    n_function_evaluations, num_iterations, yy = 0, 0, []
    a, b = bounds
    sqrt_eps, golden_mean = 1.4832396974191326e-08, 0.3819660112501051
    gm = a + golden_mean * (b - a)
    gm_1 = gm_2 = gm
    rat = e = 0.0
    y = func(gm_2)
    n_function_evaluations += 1
    yy.append(y)
    if (n_function_evaluations == max_function_evaluations) or (y < fitness_threshold):
        return y, gm_2, yy
    y_1 = y_2 = y
    middle = 0.5 * (a + b)
    tol_1 = sqrt_eps * np.abs(gm_2) + tol / 3.0
    tol_2 = 2.0 * tol_1
    while np.abs(gm_2 - middle) > (tol_2 - 0.5 * (b - a)):
        golden = 1
        if np.abs(e) > tol_1:
            golden = 0
            r = (gm_2 - gm_1) * (y - y_1)
            q = (gm_2 - gm) * (y - y_2)
            p = (gm_2 - gm) * q - (gm_2 - gm_1) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat
            if (
                (np.abs(p) < np.abs(0.5 * q * r))
                and (p > q * (a - gm_2))
                and (p < q * (b - gm_2))
            ):
                rat = (p + 0.0) / q
                x = gm_2 + rat
                if ((x - a) < tol_2) or ((b - x) < tol_2):
                    rat = tol_1 * (np.sign(middle - gm_2) + ((middle - gm_2) == 0))
            else:
                golden = 1
        if golden:
            if gm_2 >= middle:
                e = a - gm_2
            else:
                e = b - gm_2
            rat = golden_mean * e
        x = gm_2 + (np.sign(rat) + (rat == 0)) * np.maximum(np.abs(rat), tol_1)
        yyy = func(x)
        n_function_evaluations += 1
        yy.append(y)
        if (n_function_evaluations == max_function_evaluations) or (
            y < fitness_threshold
        ):
            return y, gm_2, yy
        if yyy <= y:
            if x >= gm_2:
                a = gm_2
            else:
                b = gm_2
            gm, y_1 = gm_1, y_2
            gm_1, y_2 = gm_2, y
            gm_2, y = x, yyy
        else:
            if x < gm_2:
                a = x
            else:
                b = x
            if (yyy <= y_2) or (gm_1 == gm_2):
                gm, y_1 = gm_1, y_2
                gm_1, y_2 = x, yyy
            elif (yyy <= y_1) or (gm == gm_2) or (gm == gm_1):
                gm, y_1 = x, yyy
        middle = 0.5 * (a + b)
        tol_1 = sqrt_eps * np.abs(gm_2) + tol / 3.0
        tol_2 = 2.0 * tol_1
        num_iterations += 1
        if num_iterations == max_iterations - 1:
            break
    return y, gm_2, yy


def _line_for_search(x0, alpha, lb, ub):
    # this is adopted from https://github.com/scipy/scipy/blob/main/scipy/optimize/_optimize.py
    (nonzero,) = alpha.nonzero()
    if len(nonzero) == 0:
        return 0, 0
    lb, ub = lb[nonzero], ub[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low, high = (lb - x0) / alpha, (ub - x0) / alpha
    pos = alpha > 0
    min_pos, min_neg = np.where(pos, low, 0), np.where(pos, 0, high)
    max_pos, max_neg = np.where(pos, high, 0), np.where(pos, 0, low)
    l_min, l_max = np.max(min_pos + min_neg), np.min(max_pos + max_neg)
    return (l_min, l_max) if l_max >= l_min else (0, 0)


class POWELL(DS):
    def __init__(self, problem, options):
        DS.__init__(self, problem, options)
        self._func = None  # only for inner line searcher
        self.y_history = []
        self.x_history = []

    def initialize(self, x=None, y=None, u=None, args=None, is_restart=False):
        x = (
            self._initialize_x(is_restart) if x is None else x
        )  # initial (starting) search point
        y = self._evaluate_fitness(x, args) if y is None else y  # fitness
        u = np.identity(self.ndim_problem) if u is None else u
        self.y_history.append(y)
        self.x_history.append(x)

        def _wrapper(xx):
            return self._evaluate_fitness(xx, args)

        self._func = _wrapper
        return x, y, u, y

    def _line_search(self, x, d, tol=1e-4 * 100):
        def _func(alpha):  # only for line search
            return self._func(x + alpha * d)

        bound = _line_for_search(x, d, self.lower_boundary, self.upper_boundary)
        y, gm, yy = _minimize_scalar_bounded(
            _func,
            bound,
            self.max_function_evaluations - self.n_function_evaluations,
            self.fitness_threshold,
            tol / 100.0,
        )
        d *= gm
        return y, x + d, d, yy

    def iterate(self, x=None, y=None, u=None, args=None):
        xx, yy = np.copy(x), np.copy(y)
        big_ind, delta, ys = 0, 0.0, []
        for i in range(self.ndim_problem):
            if self._check_terminations():
                return x, y, u, ys
            d, diff = u[i], y
            y, x, d, fitness = self._line_search(x, d)
            self.y_history.append(y)
            self.x_history.append(x)
            ys.extend(fitness)
            diff -= y
            if diff > delta:
                delta, big_ind = diff, i
        d = x - xx  # extrapolated point
        _, ratio_e = _line_for_search(x, d, self.lower_boundary, self.upper_boundary)
        xxx = x + min(ratio_e, 1.0) * d
        yyy = self.fitness_function(xxx)
        self.y_history.append(yyy)
        self.x_history.append(xxx)
        if yy > yyy:
            t, temp = 2.0 * (yy + yyy - 2.0 * y), yy - y - delta
            t *= np.square(temp)
            temp = yy - yyy
            t -= delta * np.square(temp)
            if t < 0.0:
                y, x, d, fitness = self._line_search(x, d)
                self.y_history.append(y)
                self.x_history.append(x)
                ys.extend(fitness)
                if np.any(d):
                    u[big_ind] = u[-1]
                    u[-1] = d
        self._n_generations += 1
        return x, y, u, ys

    def optimize(self, fitness_function=None, args=None):
        fitness = DS.optimize(self, fitness_function)
        x = self.start_conditions.get("x", None)
        y = self.start_conditions.get("y", None)
        u = self.start_conditions.get("u", None)

        x, y, u, yy = self.initialize(x=x, y=y, u=u, args=args)
        while not self.termination_signal:
            self._print_verbose_info(fitness, yy)
            x, y, u, yy = self.iterate(x, y, u, args)
        results = self._collect(fitness, yy)
        results.update({"u": u})
        return results

    def set_data(
        self,
        x=None,
        y=None,
        u=None,
        yy=None,
        *args,
        **kwargs,
    ):
        start_conditions = {i: None for i in ("x", "y", "u")}
        if x is None or y is None:
            self.start_conditions = start_conditions
            return
        if isinstance(x, np.ndarray) and len(x.shape) > 1:
            x = x.mean(axis=0)
            y = None
            u = None
        self.start_conditions = {
            i: locals().get(i, None)
            for i in (
                "x",
                "y",
                "u",
            )
        }

    def get_data(self, n_individuals: Optional[int] = None):
        pop_data = ["x", "y"]
        best_indices = sorted(
            [i for i in range(len(self.y_history))],
            key=lambda x: self.y_history[x],
        )[:n_individuals]
        x = np.array(self.x_history)[best_indices]
        y = np.array(self.y_history)[best_indices]
        return (
            self.results | {i: locals()[i] for i in pop_data} or self.start_conditions
        )
