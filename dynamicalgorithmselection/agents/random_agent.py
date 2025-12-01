import numpy as np
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class RandomAgent(Agent):
    def __init__(self, problem, options):
        Agent.__init__(self, problem, options)

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results, None

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)

        x, y = None, None
        iteration_result = {"x": x, "y": y}
        while not self._check_terminations():
            action = np.random.choice(len(self.actions))
            action_options = {k: v for k, v in self.options.items()}
            action_options["max_function_evaluations"] = min(
                self.checkpoints[self._n_generations],
                self.max_function_evaluations,
            )
            action_options["verbose"] = False
            optimizer = self.actions[action](self.problem, action_options)
            optimizer.n_function_evaluations = self.n_function_evaluations
            optimizer._n_generations = 0
            iteration_result = self.iterate(iteration_result, optimizer)
            x, y = iteration_result.get("x"), iteration_result.get("y")

            self.n_function_evaluations = optimizer.n_function_evaluations
            self._print_verbose_info(fitness, y)
            self.n_function_evaluations = optimizer.n_function_evaluations
        return self._collect(fitness, self.best_so_far_y)
