import numpy as np
import torch
import copy
import os

from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.agents.agent_state import get_la_features
from dynamicalgorithmselection.agents.ppo_utils import (
    DEVICE,
    RolloutBuffer,
    RLDASNetwork,
    GAMMA,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer

INITIAL_POPSIZE = 170


class RLDASAgent(Agent):
    def __init__(self, problem, options):
        super().__init__(problem, options)

        self.alg_names = [alg.__name__ for alg in self.actions]
        self.n_algorithms = len(self.actions)

        self.network = RLDASNetwork(
            num_algorithms=self.n_algorithms, d_dim=self.ndim_problem
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-5)

        self._load_parameters(options)
        self.ah_vectors = np.zeros((self.n_algorithms, 2, self.ndim_problem))
        self.alg_usage_counts = np.zeros(self.n_algorithms)
        self.context_memory = {name: {} for name in self.alg_names}
        self.context_memory["Common"] = {}
        self.mean_rewards = options.get("mean_rewards", [])
        self.best_50_mean = float("inf")
        self.schedule_interval = options.get(
            "schedule_interval", int(self.max_function_evaluations / 50)
        )

        expected_trajectory_length = int(
            np.ceil(self.max_function_evaluations / self.schedule_interval)
        )
        buffer_capacity = expected_trajectory_length * 10  # Safety margin
        self.buffer = RolloutBuffer(capacity=buffer_capacity, device=DEVICE)

    def _load_parameters(self, options):
        if p := options.get("network_parameters", None):
            self.network.load_state_dict(p)
        if p := options.get("optimizer", None):
            self.optimizer.load_state_dict(p)

    def get_state(self, pop_x, pop_y):
        la = get_la_features(self, pop_x, pop_y)
        ah = self.ah_vectors.copy()

        return la, ah

    def _update_ah_history(
        self, alg_idx, x_best_old, x_best_new, x_worst_old, x_worst_new
    ):
        """
        Updates Shift Vectors (SV) for the selected algorithm.
        Eq (8), (9).
        """
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
        common_attrs = ["memory_f", "memory_cr", "archive", "archive_fitness"]
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
                "successful_f",
                "successful_cr",
            ]
        elif "MadDE" in alg_name:
            specific_attrs = ["pm", "pbest", "pqBX"]
        elif "NL_SHADE" in alg_name:
            specific_attrs = ["nA", "pA"]

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

    def _select_action(self, state):
        """
        Selects action using the shared network with split inputs.
        """
        la_state, ah_state = state
        la_tensor = torch.FloatTensor(la_state).unsqueeze(0).to(DEVICE)
        ah_tensor = torch.FloatTensor(ah_state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs, value = self.network(la_tensor, ah_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            probs = probs.detach().cpu().numpy()[0]

            if self.run is not None:
                entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))
                self.run.log({"normalized entropy": entropy})

        return action.item(), log_prob, value

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
        if self.saving_fitness:
            fitness.append(self.best_so_far_y)

        self.initial_cost = best_y_global if abs(best_y_global) > 1e-8 else 1.0

        self.ah_vectors.fill(0.0)
        self.alg_usage_counts.fill(0.0)
        self.context_memory = {name: {} for name in self.alg_names}
        self.context_memory["Common"] = {}

        trajectory = []  # To store (s, a, r_raw, log_prob, val, done)

        while self.n_function_evaluations < self.max_function_evaluations:
            state = self.get_state(population_x, population_y)

            action_idx, log_prob, value = self._select_action(state)
            self.choices_history.append(action_idx)

            selected_alg_class = self.actions[action_idx]
            alg_name = self.alg_names[action_idx]

            sub_opt = selected_alg_class(self.problem, self.options)
            sub_opt.n_function_evaluations = self.n_function_evaluations
            sub_opt.max_function_evaluations = self.max_function_evaluations

            self._restore_context(sub_opt, alg_name)

            x_best_old = population_x[np.argmin(population_y)].copy()
            x_worst_old = population_x[np.argmax(population_y)].copy()
            cost_old = np.copy(np.min(population_y))

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

            x_best_new = population_x[np.argmin(population_y)].copy()
            x_worst_new = population_x[np.argmax(population_y)].copy()
            cost_new = np.min(population_y)

            self._update_ah_history(
                action_idx, x_best_old, x_best_new, x_worst_old, x_worst_new
            )

            adc = (cost_old - cost_new) / self.initial_cost
            if self.run:
                self.run.log({"adc": adc})

            done = self.n_function_evaluations >= self.max_function_evaluations

            trajectory.append(
                {
                    "state": state,
                    "action": action_idx,
                    "adc": adc,
                    "log_prob": log_prob,
                    "value": value,
                    "done": done,
                }
            )

            best_y_global = min(best_y_global, cost_new)

            # Update Agent Best State and History
            if cost_new < self.best_so_far_y:
                self.best_so_far_y = cost_new
                self.best_so_far_x = x_best_new

            self.history.append(self.best_so_far_y)
            if self.saving_fitness:
                fitness.append(self.best_so_far_y)

            self._n_generations += 1
            self._print_verbose_info(fitness, self.best_so_far_y)

        fes_end = self.n_function_evaluations
        speed_factor = self.max_function_evaluations / fes_end

        for step in trajectory:
            final_reward = step["adc"] * speed_factor
            self.rewards.append(final_reward)
            la_state, ah_state = step["state"]

            la_tensor = torch.FloatTensor(la_state).to(DEVICE)
            ah_tensor = torch.FloatTensor(ah_state).to(DEVICE)

            self.buffer.add(
                (la_tensor, ah_tensor),
                step["action"],
                final_reward,
                step["done"],
                step["log_prob"],
                step["value"],
            )

        if self.train_mode:
            T = len(trajectory)
            K = max(1, int(0.3 * T))

            self.ppo_update(
                self.buffer,
                epochs=K,
                minibatch_size=32,
                clip_eps=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
            )

            self.buffer.clear()

        return self._collect(fitness, self.best_so_far_y)

    def _collect(self, fitness, y=None):
        results, _ = super()._collect(fitness, y)
        self.mean_rewards.append(sum(self.rewards) / len(self.rewards))
        agent_state = {
            "network_parameters": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer,
            "mean_rewards": self.mean_rewards,
            "reward_normalizer": self.reward_normalizer,
            "state_normalizer": self.state_normalizer,
        }

        last_50_mean = sum(self.mean_rewards[-50:]) / len(self.mean_rewards[-50:])
        if self.best_50_mean > last_50_mean:
            self.best_50_mean = last_50_mean
            torch.save(agent_state, os.path.join("models", f"{self.name}_best.pth"))

        if self.n_function_evaluations == self.max_function_evaluations:
            torch.save(agent_state, os.path.join("models", f"{self.name}_final.pth"))

        return results, agent_state

    def ppo_update(
        self,
        buffer,
        epochs=4,
        minibatch_size=None,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        la_list, ah_list = zip(*buffer.states)

        la_states = torch.stack(la_list).to(DEVICE)
        ah_states = torch.stack(ah_list).to(DEVICE)

        actions = torch.tensor(buffer.actions).to(DEVICE)
        rewards = buffer.rewards
        dones = buffer.dones

        old_logprobs = torch.stack(buffer.log_probs).detach().to(DEVICE).view(-1)

        old_values = torch.stack(buffer.values).detach().to(DEVICE).view(-1)

        for _ in range(epochs):
            policy_probs, current_values = self.network(la_states, ah_states)

            dist = torch.distributions.Categorical(policy_probs)
            logprobs = dist.log_prob(actions)

            current_values = current_values.squeeze()

            with torch.no_grad():
                if dones[-1]:
                    R = 0.0
                else:
                    R = current_values[-1].item() if not dones[-1] else 0.0

            Returns = []
            for r in reversed(rewards):
                R = R * GAMMA + r
                Returns.insert(0, R)

            Returns = torch.tensor(Returns).to(DEVICE).float()
            advantages = Returns - current_values.detach()
            ratios = torch.exp(logprobs - old_logprobs)

            # Actor Loss (Reinforce Loss with Clipping)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            vpredclipped = old_values + torch.clamp(
                current_values - old_values, -clip_eps, clip_eps
            )

            v_max = torch.max(
                ((current_values - Returns) ** 2), ((vpredclipped - Returns) ** 2)
            )
            critic_loss = v_max.mean()

            loss = critic_loss + actor_loss

            if self.run is not None:
                self.run.log(
                    {
                        "Returns": Returns.mean().item(),
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                    }
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
