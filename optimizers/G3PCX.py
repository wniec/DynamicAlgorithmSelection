import time
from operator import itemgetter

import numpy as np  # engine for numerical computing

from optimizers.Optimizer import Optimizer


class G3PCX(Optimizer):

    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        if self.n_individuals is None:  # population size
            self.n_individuals = 100
        assert self.n_individuals > 0
        self._n_generations = 0
        self.n_offsprings = options.get('n_offsprings', 2)
        assert self.n_offsprings > 0
        self.n_parents = options.get('n_parents', 3)
        assert self.n_parents > 0
        self._std_pcx_1 = options.get('_std_pcx_1', 0.1)
        self._std_pcx_2 = options.get('_std_pcx_2', 0.1)
        self._elitist = None  # index of elitist

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def initialize(self, args=None, x=None, y=None):
        x = x if x is not None else self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                            size=(self.n_individuals, self.ndim_problem))  # population
        y = y if y is not None else np.empty((self.n_individuals,))  # fitness
        for i in range(self.n_individuals):
            if self._check_terminations():
                break
            y[i] = self._evaluate_fitness(x[i], args)
        return x, y

    def iterate(self, x=None, y=None, args=None):
        self._elitist, fitness = np.argmin(y), []
        # (Step 1:) select the best and `self.n_parents - 1` other parents randomly from the population
        parents = self.rng_optimization.choice(self.n_individuals, size=self.n_parents, replace=False)
        if self._elitist not in parents:  # to ensure that elitist is always included
            parents[0] = self._elitist
        # (Step 2:) generate offspring from the chosen parents using a recombination scheme
        xx, yy = np.empty((self.n_offsprings, self.ndim_problem)), np.empty((self.n_offsprings,))
        g = np.mean(x[parents], axis=0)  # mean vector of the chosen parents
        for i in range(self.n_offsprings):
            if self._check_terminations():
                break
            p = self._elitist  # for faster local convergence
            d = g - x[p]
            d_norm = np.linalg.norm(d)
            d_mean = np.empty((self.n_parents - 1,))
            diff = np.empty((self.n_parents - 1, self.ndim_problem))  # for distance computation
            for ii, j in enumerate(parents[1:]):
                diff[ii] = x[j] - x[p]  # distance from one parent
            for ii in range(self.n_parents - 1):
                d_mean[ii] = np.linalg.norm(diff[ii]) * np.sqrt(
                    1.0 - np.power(np.dot(diff[ii], d) / (np.linalg.norm(diff[ii]) * d_norm), 2))
            d_mean = np.mean(d_mean)  # average of perpendicular distances
            orth = self._std_pcx_2 * d_mean * self.rng_optimization.standard_normal((self.ndim_problem,))
            orth = orth - (np.dot(orth, d) * d) / np.power(d_norm, 2)
            xx[i] = x[p] + self._std_pcx_1 * self.rng_optimization.standard_normal() * d + orth
            yy[i] = self._evaluate_fitness(xx[i], args)
            fitness.append(yy[i])
        # (Step 3:) choose two parents at random from the population
        offsprings = self.rng_optimization.choice(self.n_individuals, size=2, replace=False)
        # (Step 4:) from a combined subpopulation of two chosen parents and created offspring, choose
        #   the best two solutions and replace the chosen two parents (in Step 3) with these solutions
        xx, yy = np.vstack((xx, x[offsprings])), np.hstack((yy, y[offsprings]))
        x[offsprings], y[offsprings] = xx[np.argsort(yy)[:2]], yy[np.argsort(yy)[:2]]
        self.results['x'] = x
        self.results['y'] = y
        self._n_generations += 1
        return fitness

    def _collect(self, fitness, y=None):
        self._print_verbose_info(fitness, y)
        for i in range(1, len(fitness)):  # to avoid `np.nan`
            if np.isnan(fitness[i]):
                fitness[i] = fitness[i - 1]
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        x, y = itemgetter('x', 'y')(self.start_conditions)
        x, y = self.initialize(args, x, y)
        yy = y  # only for printing
        while not self._check_terminations():
            self._print_verbose_info(fitness, yy)
            yy = self.iterate(x, y, args)
        return self._collect(fitness, yy)