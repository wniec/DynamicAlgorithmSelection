import numpy as np
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class RandomAgent(Agent):
    def __init__(self, problem, options):
        Agent.__init__(self, problem, options)
        self.iterations_history = {"x": None, "y": None}

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results, None

    def _select_action(self):
        probs = np.ones_like(self.actions, dtype=float) / len(self.actions)
        action = np.random.choice(len(probs), p=probs)
        return action

    def _execute_action(self, action_idx, iteration_result):
        action_options = {k: v for k, v in self.options.items()}
        action_options["max_function_evaluations"] = min(
            self.checkpoints[self._n_generations],
            self.max_function_evaluations,
        )
        action_options["verbose"] = False
        optimizer = self.actions[action_idx](self.problem, action_options)
        optimizer.n_function_evaluations = self.n_function_evaluations
        optimizer._n_generations = 0
        return self.iterate(iteration_result, optimizer), optimizer

    def _update_history(self, iteration_result):
        for key, val in iteration_result.items():
            if key.endswith("_history") and key != "fitness_history":
                variable_name = key[: -len("_history")]
                appended_val = iteration_result.get(key)
                historic_val = self.iterations_history.get(variable_name)

                if historic_val is None:
                    self.iterations_history[variable_name] = appended_val
                elif appended_val.shape != (0,):
                    self.iterations_history[variable_name] = np.concatenate(
                        (historic_val, appended_val)
                    )

        return iteration_result

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y = None, None
        self.iterations_history = {"x": None, "y": None}
        iteration_result = {"x": x, "y": y}
        last_used_params = []
        while True:
            action = self._select_action()
            iteration_result, optimizer = self._execute_action(action, iteration_result)
            if len(last_used_params) > 0:
                for key in last_used_params:
                    if key in self.iterations_history and key not in iteration_result:
                        self.iterations_history.pop(key)
            last_used_params = optimizer.start_condition_parameters
            _, y = iteration_result.get("x"), iteration_result.get("y")
            iteration_result = self._update_history(iteration_result)
            self.n_function_evaluations = optimizer.n_function_evaluations
            self._print_verbose_info(fitness, y)
            self.n_function_evaluations = optimizer.n_function_evaluations
            if self._check_terminations() or self._n_generations == self.n_checkpoints:
                break
        return self._collect(fitness, self.best_so_far_y)
