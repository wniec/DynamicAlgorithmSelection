import numpy as np
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class NeuroevolutionAgent(Agent):
    def __init__(self, problem, options):
        Agent.__init__(self, problem, options)
        self.net = options["net"]

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results, _ = super()._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        results["mean_reward"] = sum(self.rewards) / len(self.rewards)
        results["actions"] = self.choices_history
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y, reward = None, None, None
        iteration_result = {"x": x, "y": y}
        while not self._check_terminations():
            state = self.get_state(x, y)
            state = np.nan_to_num(state, nan=0.5, neginf=0.0, posinf=1.0)
            policy = self.net.activate(state)
            probs = np.array(policy)
            probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
            probs /= probs.sum()

            action = np.random.choice(len(probs), p=probs)
            self.choices_history.append(action)
            action_options = {k: v for k, v in self.options.items()}
            action_options["max_function_evaluations"] = min(
                self.checkpoints[self._n_generations],
                self.max_function_evaluations,
            )
            action_options["verbose"] = False
            optimizer = self.actions[action](self.problem, action_options)
            optimizer.n_function_evaluations = self.n_function_evaluations
            optimizer._n_generations = 0
            best_parent = self.best_so_far_y
            iteration_result = self.iterate(iteration_result, optimizer)
            x, y = iteration_result.get("x"), iteration_result.get("y")

            new_best_y = self.best_so_far_y
            reward = self.get_reward(new_best_y, best_parent)
            self.rewards.append(reward)
            if self.run:
                self.run.log({"reward": reward})

            self.n_function_evaluations = optimizer.n_function_evaluations
            if self.train_mode:
                pass
            self._print_verbose_info(fitness, y)
            if optimizer.best_so_far_y >= self.best_so_far_y:
                self.stagnation_count += (
                    optimizer.n_function_evaluations - self.n_function_evaluations
                )
            else:
                self.stagnation_count = 0

            self.n_function_evaluations = optimizer.n_function_evaluations
        return self._collect(fitness, self.best_so_far_y)
