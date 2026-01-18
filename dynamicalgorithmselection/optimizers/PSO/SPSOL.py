import numpy as np
from dynamicalgorithmselection.optimizers.PSO.PSO import PSO


class SPSOL(PSO):
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)

    def _ring_topology(self, p_x=None, p_y=None, i=None):
        left, right = i - 1, i + 1
        if i == 0:
            left = self.n_individuals - 1
        elif i == self.n_individuals - 1:
            right = 0
        ring = [left, i, right]
        return p_x[ring[int(np.argmin(p_y[ring]))]]

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = self._ring_topology(
                p_x, p_y, i
            )  # online update within ring topology
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = (
                self._w[min(self._n_generations, len(self._w))] * v[i]
                + self.cognition * cognition_rand * (p_x[i] - x[i])
                + self.society * society_rand * (n_x[i] - x[i])
            )  # velocity update
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]  # position update
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i], args)  # fitness evaluation
            if y[i] < p_y[i]:  # online update
                p_x[i], p_y[i] = x[i], y[i]
        self._n_generations += 1
        self.results.update(
            {"v": v, "x": x, "y": y, "p_x": p_x, "p_y": p_y, "n_x": n_x}
        )
        return v, x, y, p_x, p_y, n_x
