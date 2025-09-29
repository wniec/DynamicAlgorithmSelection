import time
from operator import itemgetter

import numpy as np

from optimizers.G3PCX import G3PCX
from optimizers.LMCMAES import LMCMAES
from optimizers.Optimizer import Optimizer
from optimizers.SPSO import SPSO
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.i = 0

    def forward(self, x):
        self.i += 1
        return self.i % 3


class Agent(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self._n_generations = 0
        self.n_individuals = 0
        self.problem = problem
        self.options = options
        sub_optimization_ratio = options['sub_optimization_ratio']
        self.sub_optimizer_max_fe = self.max_function_evaluations / sub_optimization_ratio

        self.ACTIONS = [G3PCX,
                        SPSO,
                        LMCMAES]

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and ((not self._n_generations % self.verbose) or (self.termination_signal > 0)):
            info = '  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}'
            print(info.format(self._n_generations, self.best_so_far_y, np.min(y), self.n_function_evaluations))

    def _save_fitness(self, x, y):
        # update best-so-far solution (x) and fitness (y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        # update all settings related to early stopping
        if (self._base_early_stopping - y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, y

    def iterate(self, x=None, y=None, optimizer=None):
        optimizer.set_data(x, y)
        if self._check_terminations():
            return x, y
        self._n_generations += 1
        results = optimizer.optimize()
        self._save_fitness(results['best_so_far_x'], results['best_so_far_y'])  # fitness evaluation
        return itemgetter('x', 'y')(optimizer.get_data())

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results['_n_generations'] = self._n_generations
        return results

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        state = 0
        model = Model()
        x, y = None, None
        while not self._check_terminations():
            action = model(state)
            action_options = {k: v for k, v in self.options.items()}
            action_options['max_function_evaluations'] = min(self.n_function_evaluations + self.sub_optimizer_max_fe, self.max_function_evaluations)
            action_options['verbose'] = False
            optimizer = self.ACTIONS[action](self.problem, action_options)
            optimizer.n_function_evaluations = self.n_function_evaluations
            optimizer._n_generations = 0
            x, y = self.iterate(x, y, optimizer)
            self._print_verbose_info(fitness, y)
            self.n_function_evaluations = optimizer.n_function_evaluations
        return self._collect(fitness, y)
