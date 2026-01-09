import numpy as np
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.agents.agent_utils import MAX_DIM
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
        results.update(
            {
                "reward_normalizer": self.reward_normalizer,
            }
        )
        return results

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)
        x, y, reward = None, None, None
        iteration_result = {"x": x, "y": y}
        x_history, y_history = None, None
        step_idx = 0
        while not self._check_terminations():
            used_fe = self.n_function_evaluations / self.max_function_evaluations
            dim_coef = self.ndim_problem / MAX_DIM
            if x is not None and y is not None:
                x, y = x.astype(np.float32), y.astype(np.float32)
            state = (
                self.get_state(x_history, y_history).flatten()
                if self.options.get("state_representation") == "ELA"
                else self.get_state(x, y).flatten()
            )
            state = np.append(state, (used_fe, dim_coef))
            state = np.nan_to_num(state, nan=0.5, neginf=0.0, posinf=1.0)
            policy = self.net.activate(state)
            action = np.argmax(policy)
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
            x_history, y_history = (
                iteration_result.get("x_history"),
                iteration_result.get("y_history"),
            )

            new_best_y = self.best_so_far_y
            reward = self.get_reward(new_best_y, best_parent)
            # reward = self.reward_normalizer.normalize(reward, step_idx)
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
            step_idx += 1
        return self._collect(fitness, self.best_so_far_y)
