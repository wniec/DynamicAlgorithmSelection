import numpy as np

from dynamicalgorithmselection.optimizers.GA.GA import GA
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class G3PCX(GA):
    def __init__(self, problem, options):
        GA.__init__(self, problem, options)
        self.n_offsprings = options.get("n_offsprings", 2)
        assert self.n_offsprings > 0
        self.n_parents = options.get("n_parents", 3)
        assert self.n_parents > 0
        self._std_pcx_1 = options.get("_std_pcx_1", 0.1)
        self._std_pcx_2 = options.get("_std_pcx_2", 0.1)
        self._elitist = None  # index of elitist

    def initialize(self, args=None, x=None, y=None):
        recalculate_y = y is None
        x = (
            x
            if x is not None
            else self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                size=(self.n_individuals, self.ndim_problem),
            )
        )  # population
        y = np.empty((self.n_individuals,)) if recalculate_y else y  # fitness
        if recalculate_y:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    break
                y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def iterate(self, x=None, y=None, args=None):
        self._elitist, fitness = np.argmin(y), []
        # (Step 1:) select the best and `self.n_parents - 1` other parents randomly from the population
        parents = self.rng_optimization.choice(
            self.n_individuals, size=self.n_parents, replace=False
        )
        if self._elitist not in parents:  # to ensure that elitist is always included
            parents[0] = self._elitist
        # (Step 2:) generate offspring from the chosen parents using a recombination scheme
        xx, yy = (
            np.empty((self.n_offsprings, self.ndim_problem)),
            np.empty((self.n_offsprings,)),
        )
        g = np.mean(x[parents], axis=0)  # mean vector of the chosen parents
        for i in range(self.n_offsprings):
            if self._check_terminations():
                break
            p = self._elitist  # for faster local convergence
            d = g - x[p]
            d_norm = np.linalg.norm(d)
            d_mean = np.empty((self.n_parents - 1,))
            diff = np.empty(
                (self.n_parents - 1, self.ndim_problem)
            )  # for distance computation
            for ii, j in enumerate(parents[1:]):
                diff[ii] = x[j] - x[p]  # distance from one parent
            for ii in range(self.n_parents - 1):
                # added 1e-8 for numerical stability, made sqrt from positive only
                d_mean[ii] = np.linalg.norm(diff[ii]) * np.sqrt(
                    max(
                        1.0
                        - np.power(
                            np.dot(diff[ii], d)
                            / ((np.linalg.norm(diff[ii])) * d_norm + 1e-8),
                            2,
                        ),
                        0,
                    )
                )
            d_mean = np.mean(d_mean)  # average of perpendicular distances
            orth = (
                self._std_pcx_2
                * d_mean
                * self.rng_optimization.standard_normal((self.ndim_problem,))
            )
            orth = orth - (np.dot(orth, d) * d) / (np.power(d_norm, 2) + 1e-8)
            xx[i] = (
                x[p]
                + self._std_pcx_1 * self.rng_optimization.standard_normal() * d
                + orth
            )
            yy[i] = self._evaluate_fitness(xx[i], args)
            fitness.append(yy[i])
        # (Step 3:) choose two parents at random from the population
        offsprings = self.rng_optimization.choice(
            self.n_individuals, size=2, replace=False
        )
        # (Step 4:) from a combined subpopulation of two chosen parents and created offspring, choose
        #   the best two solutions and replace the chosen two parents (in Step 3) with these solutions
        xx, yy = np.vstack((xx, x[offsprings])), np.hstack((yy, y[offsprings]))
        x[offsprings], y[offsprings] = xx[np.argsort(yy)[:2]], yy[np.argsort(yy)[:2]]
        self.results["x"] = x
        self.results["y"] = y
        self._n_generations += 1
        return fitness

    def _collect(self, fitness, y=None):
        self._print_verbose_info(fitness, y)
        for i in range(1, len(fitness)):  # to avoid `np.nan`
            if np.isnan(fitness[i]):
                fitness[i] = fitness[i - 1]
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = GA.optimize(self, fitness_function)
        x, y = (
            self.start_conditions.get("x", None),
            self.start_conditions.get("y", None),
        )
        x, y = self.initialize(args, x, y)
        yy = y  # only for printing
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            yy = self.iterate(x, y, args)
        return self._collect(fitness, yy)
