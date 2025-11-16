import numpy as np
import torch

from dynamicalgorithmselection.agents.agent_state import AgentState

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class Agent(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.rewards = []
        self.choices_history = []
        self.stagnation_count = 0
        self._n_generations = 0
        self.problem = problem
        self.rewards = []
        self.options = options
        self.history = []
        self.actions = options.get("action_space")

        self.train_mode = options.get("train_mode", True)

        sub_optimization_ratio = options["sub_optimization_ratio"]
        self.run = options.get("run", None)
        self.sub_optimizer_max_fe = (
            self.max_function_evaluations / sub_optimization_ratio
        )

    def get_initial_state(self):
        vector = [
            0.0,  # third weighted central moment
            0.0,  # second weighted central moment
            0.0,  # normalized domination of best solution
            0.0,  # normalized radius of the smallest sphere containing entire population
            0.5,  # normalized relative fitness difference
            0.5,  # average_y relative to best
            1.0,  # normalized y deviation measure
            1.0,  # full remaining budget (max evaluations)
            self.ndim_problem / 40,  # normalized problem dimension
            0.0,  # stagnation count
            *([0.0] * (49 + 2 * len(self.actions))),
        ]
        return torch.tensor(vector, dtype=torch.float)

    def get_state(self, x: np.ndarray, y: np.ndarray) -> np.array:
        if x is None or y is None:
            return self.get_initial_state()
        else:
            state = AgentState(
                x,
                y,
                self.best_so_far_x,
                self.best_so_far_y,
                self.lower_boundary,
                self.upper_boundary,
                self.worst_so_far_x,
                self.worst_so_far_y,
                self.history,
                self.choices_history,
                len(self.actions),
            )
            used_fe_ratio = self.max_function_evaluations / self.sub_optimizer_max_fe

            vector = [
                state.get_weighted_central_moment(3),
                state.get_weighted_central_moment(2),
                state.mean_falling_behind(),
                state.population_relative_radius(),
                state.relative_improvement(),
                state.y_historic_improvement(),
                state.y_deviation(),
                1 - used_fe_ratio,
                self.ndim_problem
                / 40,  # maximum dimensionality in this COCO benchmark is 40
                self.stagnation_count / self.max_function_evaluations,
                *state.distances_from_best(),
                *state.distances_from_mean(),
                *state.relative_y_differences(),
                *state.last_action_encoded,
                state.same_action_counter() / used_fe_ratio,
                *state.choices_frequency,
                state.explored_volume() ** (1 / self.ndim_problem),  # searched volume
                *state.x_standard_deviation_stats(),
                *state.normalized_x_stats(),
                state.choice_entropy(),
                state.normalized_distance(self.best_so_far_x, self.worst_so_far_x),
                *state.y_difference_stats(),
                *state.slopes_stats(),
            ]
        return torch.tensor(vector, dtype=torch.float)

    def _print_verbose_info(self, fitness, y):
        if self.saving_fitness:
            if not np.isscalar(y):
                fitness.extend(y)
            else:
                fitness.append(y)
        if self.verbose and (
            (not self._n_generations % self.verbose) or (self.termination_signal > 0)
        ):
            info = "  * Generation {:d}: best_so_far_y {:7.5e}, min(y) {:7.5e} & Evaluations {:d}"
            print(
                info.format(
                    self._n_generations,
                    self.best_so_far_y,
                    np.min(y),
                    self.n_function_evaluations,
                )
            )

    def _save_fitness(self, best_x, best_y, worst_x, worst_y):
        self.history.append(best_y)
        if best_y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(best_x), best_y
        if worst_y > self.worst_so_far_y:
            self.worst_so_far_x, self.worst_so_far_y = np.copy(worst_x), worst_y
        # update all settings related to early stopping
        if (self._base_early_stopping - best_y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, best_y

    def iterate(self, optimizer_input_data=None, optimizer=None):
        optimizer_input_data["best_x"] = self.best_so_far_x
        optimizer_input_data["best_y"] = self.best_so_far_y
        optimizer.set_data(**optimizer_input_data)
        if self._check_terminations():
            return optimizer.get_data()
        self._n_generations += 1
        results = optimizer.optimize()
        self._save_fitness(
            results["best_so_far_x"],
            results["best_so_far_y"],
            results["worst_so_far_x"],
            results["worst_so_far_y"],
        )  # fitness evaluation
        return optimizer.get_data(self.n_individuals)

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        if self.run:
            choices_count = {
                self.actions[j].__name__: sum(1 for i in self.choices_history if i == j)
                / (len(self.choices_history) or 1)
                for j in range(len(self.actions))
            }
            self.run.log(choices_count)
            self.run.log(
                {f"{k}_dim_{self.ndim_problem}": v for k, v in choices_count.items()},
            )
        return results, None

    def optimize(self, fitness_function=None, args=None):
        raise NotImplementedError

    def get_reward(self, y, best_parent):
        log_scale = lambda x: np.log(np.clip(x, a_min=0, a_max=None) + 1)
        reference = max(
            self.worst_so_far_y
            - (self.best_so_far_y if best_parent == float("inf") else best_parent),
            1e-5,
        )
        best_individual = np.min(y)
        improvement = (
            (best_parent - best_individual) if best_individual is not None else 0
        )
        # used_fe = self.n_function_evaluations / self.max_function_evaluations
        reward = log_scale(improvement) / log_scale(reference)
        if len(self.choices_history) > 1:
            pass
            # reward += 0.05 if self.choices_history[-1] == self.choices_history[-2] else 0.0
        else:
            return 0
        # reward = np.sign(improvement)#  * used_fe
        return np.clip(np.cbrt(reward), a_min=-0.0, a_max=0.5)  # to the 1/dim power ?
