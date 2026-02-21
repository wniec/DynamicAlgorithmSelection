import numpy as np
import torch
import copy
from typing import Any, Dict

from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer

INITIAL_POPSIZE = 170


class RLDASRandomAgent(Agent):
    def __init__(self, problem, options):
        super().__init__(problem, options)

        self.alg_names = [alg.__name__ for alg in self.actions]
        self.n_algorithms = len(self.actions)

        self.ah_vectors = np.zeros((self.n_algorithms, 2, self.ndim_problem))
        self.alg_usage_counts = np.zeros(self.n_algorithms)
        self.context_memory: Dict[str, Dict[str, Any]] = {
            name: {} for name in self.alg_names
        }
        self.context_memory["Common"] = {}
        self.mean_rewards = options.get("mean_rewards", [])
        self.best_50_mean = float("inf")
        self.schedule_interval = options.get(
            "schedule_interval", int(self.max_function_evaluations / 50)
        )

    def _update_ah_history(
        self, alg_idx, x_best_old, x_best_new, x_worst_old, x_worst_new
    ):
        sv_best_current = x_best_new - x_best_old
        sv_worst_current = x_worst_new - x_worst_old

        H = self.alg_usage_counts[alg_idx]

        self.ah_vectors[alg_idx, 0] = (
            self.ah_vectors[alg_idx, 0] * H + sv_best_current
        ) / (H + 1)
        self.ah_vectors[alg_idx, 1] = (
            self.ah_vectors[alg_idx, 1] * H + sv_worst_current
        ) / (H + 1)

        self.alg_usage_counts[alg_idx] += 1

    def _save_context(self, optimizer, alg_name):
        common_attrs = ["MF", "MCr", "archive"]
        for attr in common_attrs:
            if hasattr(optimizer, attr):
                self.context_memory["Common"][attr] = getattr(optimizer, attr)

        specific_attrs = []
        if "JDE21" in alg_name:
            specific_attrs = [
                "tau1",
                "tau2",
                "ageLmt",
                "eps",
                "myEqs",
            ]
        elif "MadDE" in alg_name:
            specific_attrs = ["pm", "pbest", "PqBX"]
        elif "NL_SHADE" in alg_name:
            specific_attrs = ["NA", "pa"]

        for attr in specific_attrs:
            if hasattr(optimizer, attr):
                self.context_memory[alg_name][attr] = getattr(optimizer, attr)

    def _restore_context(self, optimizer, alg_name):
        """
        Restores parameters to the optimizer from self.context_memory.
        """
        for attr, val in self.context_memory["Common"].items():
            if hasattr(optimizer, attr):
                setattr(optimizer, attr, copy.deepcopy(val))

        if alg_name in self.context_memory:
            for attr, val in self.context_memory[alg_name].items():
                if hasattr(optimizer, attr):
                    setattr(optimizer, attr, copy.deepcopy(val))

    def _select_action(self):
        with torch.no_grad():
            probs = torch.ones(size=(1, len(self.actions))) / len(self.actions)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action.item()

    def initialize(self):
        x = self.rng_initialization.uniform(
            self.initial_lower_boundary,
            self.initial_upper_boundary,
            size=(INITIAL_POPSIZE, self.ndim_problem),
        )
        y = np.zeros((INITIAL_POPSIZE,))
        for i in range(INITIAL_POPSIZE):
            y[i] = self._evaluate_fitness(x[i])
        return x, y

    def optimize(self, fitness_function=None, args=None):
        """
        Main Optimization Loop implementing RL-DAS workflow (Algorithm 1).
        Does NOT use checkpoints. Uses interval-based scheduling.
        """
        fitness = Optimizer.optimize(self, fitness_function)
        population_x, population_y = self.initialize()
        self.n_function_evaluations = INITIAL_POPSIZE

        best_idx = np.argmin(population_y)
        best_y_global = population_y[best_idx]
        best_x_global = population_x[best_idx].copy()

        self.best_so_far_y = best_y_global
        self.best_so_far_x = best_x_global

        self.history.append(self.best_so_far_y)
        fitness.append(float(self.best_so_far_y))

        self.ah_vectors.fill(0.0)
        self.alg_usage_counts.fill(0.0)
        self.context_memory = {name: {} for name in self.alg_names}
        self.context_memory["Common"] = {}

        while self.n_function_evaluations < self.max_function_evaluations:
            action_idx = self._select_action()
            self.choices_history.append(action_idx)

            selected_alg_class = self.actions[action_idx]
            alg_name = self.alg_names[action_idx]

            sub_opt = selected_alg_class(self.problem, self.options)
            sub_opt.n_function_evaluations = self.n_function_evaluations
            sub_opt.max_function_evaluations = self.max_function_evaluations

            self._restore_context(sub_opt, alg_name)

            x_best_old = population_x[np.argmin(population_y)].copy()
            x_worst_old = population_x[np.argmax(population_y)].copy()

            target_fes = min(
                self.n_function_evaluations + self.schedule_interval,
                self.max_function_evaluations,
            )
            sub_opt.target_FE = target_fes
            sub_opt.set_data(
                x=population_x,
                y=population_y,
                best_x=self.best_so_far_x,
                best_y=self.best_so_far_y,
            )

            res = sub_opt.optimize()

            population_x = res["x"]
            population_y = res["y"]

            self.n_function_evaluations = sub_opt.n_function_evaluations

            self._save_context(sub_opt, alg_name)

            x_best_new: np.ndarray = population_x[np.argmin(population_y)].copy()
            x_worst_new: np.ndarray = population_x[np.argmax(population_y)].copy()
            cost_new: float = np.min(population_y)

            self._update_ah_history(
                action_idx, x_best_old, x_best_new, x_worst_old, x_worst_new
            )

            best_y_global = min(best_y_global, cost_new)

            if cost_new < self.best_so_far_y:
                self.best_so_far_y = cost_new
                self.best_so_far_x = x_best_new

            self.history.append(self.best_so_far_y)
            fitness.append(float(self.best_so_far_y))

            self._n_generations += 1
            self._print_verbose_info(fitness, self.best_so_far_y)

        return self._collect(fitness, self.best_so_far_y)

    def _collect(self, fitness, y=None):
        results, _ = super()._collect(fitness, y)
        agent_state = {}
        return results, agent_state
