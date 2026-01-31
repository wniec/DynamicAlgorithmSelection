import numpy as np
from dynamicalgorithmselection.optimizers.PSO.PSO import PSO


class SPSO(PSO):
    def __init__(self, problem, options):
        PSO.__init__(self, problem, options)

    def iterate(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None, args=None):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            n_x[i] = p_x[np.argmin(p_y)]
            cognition_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            society_rand = self.rng_optimization.uniform(size=(self.ndim_problem,))
            v[i] = (
                self._w[min(self._n_generations, len(self._w) - 1)] * v[i]
                + self.cognition * cognition_rand * (p_x[i] - x[i])
                + self.society * society_rand * (n_x[i] - x[i])
            )
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)

            y[i] = self._evaluate_fitness(
                x[i], args, v=v[i], p_x=p_x[i], p_y=p_y[i], n_x=n_x[i]
            )

            if y[i] < p_y[i]:
                p_x[i], p_y[i] = x[i], y[i]
        self._n_generations += 1
        self.results.update(
            {"v": v, "x": x, "y": y, "p_x": p_x, "p_y": p_y, "n_x": n_x}
        )
        return v, x, y, p_x, p_y, n_x
