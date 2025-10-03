import time
from operator import itemgetter

import numpy as np
import torch
from matplotlib import pyplot as plt

from optimizers.G3PCX import G3PCX
from optimizers.LMCMAES import LMCMAES
from optimizers.Optimizer import Optimizer
from optimizers.SPSO import SPSO
from torch import nn

DISCOUNT_FACTOR = 0.9
device = "cuda" if torch.cuda.is_available() else ""

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(17, 20), nn.ReLU(), nn.LayerNorm(20), nn.Linear(20, 3), nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.actor(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(17, 20),
            nn.ReLU(),
            nn.LayerNorm(20),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.critic(x)


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, advantage, log_prob):
        return (-advantage * log_prob).mean()


class Agent(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.actor_losses = []
        self.critic_losses = []
        self.last_choice = None
        self.stagnation_count = 0
        self._n_generations = 0
        self.problem = problem
        self.options = options
        self.history = []
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.actor_loss_fn = ActorLoss().to(device)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-4)
        self.train_mode = options.get("train_mode", True)
        if p := options.get("actor_parameters"):
            self.actor.load_state_dict(p)
        if p := options.get("critic_parameters"):
            self.critic.load_state_dict(p)
        if p := options.get("actor_optimizer"):
            self.actor_optimizer.load_state_dict(p)
        if p := options.get("critic_optimizer"):
            self.critic_optimizer.load_state_dict(p)
        sub_optimization_ratio = options["sub_optimization_ratio"]
        self.sub_optimizer_max_fe = (
            self.max_function_evaluations / sub_optimization_ratio
        )

        self.ACTIONS = [G3PCX, SPSO, LMCMAES]

    def get_state(self, x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        # Definition of state is inspired by its representation in DE-DDQN article
        if x is None or y is None:
            return torch.zeros(size=(17,))
        dist = lambda x1, x2: np.linalg.norm(x1 - x2)
        dist_max = dist(self.lower_boundary, self.upper_boundary)
        average_y = sum(self.history) / (len(self.history) or 1)
        mid_so_far = (self.worst_so_far_y - self.best_so_far_y) / 2
        vector = [
            (np.argmin(y) - self.best_so_far_y)
            / ((self.worst_so_far_y - self.best_so_far_y) or 1),
            (average_y - self.best_so_far_y)
            / ((self.worst_so_far_y - self.best_so_far_y) or 1),
            sum((i - average_y) ** 2 for i in self.history)
            / (self.best_so_far_y - mid_so_far) ** 2
            / len(self.history),
            (self.max_function_evaluations - self.n_function_evaluations)
            / self.max_function_evaluations,
            self.ndim_problem
            / 40,  # maximum dimensionality in this COCO benchmark is 40
            self.stagnation_count / self.max_function_evaluations,
            *(
                dist(x[i], self.best_so_far_x) / dist_max
                for i in np.random.choice(self.n_individuals, 5)
            ),
            (dist(x[np.argmin(y)], self.best_so_far_x) / dist_max),
            *(
                (np.argmin(y) - y[i])
                / ((self.worst_so_far_y - self.best_so_far_y) or 1)
                for i in np.random.choice(self.n_individuals, 5)
            ),
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

    def _save_fitness(self, x, y):
        # update best-so-far solution (x) and fitness (y)
        self.history.append(y)
        if y < self.best_so_far_y:
            self.best_so_far_x, self.best_so_far_y = np.copy(x), y
        if y > self.worst_so_far_y:
            self.worst_so_far_x, self.worst_so_far_y = np.copy(x), y
        # update all settings related to early stopping
        if (self._base_early_stopping - y) <= self.early_stopping_threshold:
            self._counter_early_stopping += 1
        else:
            self._counter_early_stopping, self._base_early_stopping = 0, y

    def iterate(self, x=None, y=None, optimizer=None):
        optimizer.set_data(x, y, self.best_so_far_x, self.best_so_far_y)
        if self._check_terminations():
            return x, y
        self._n_generations += 1
        results = optimizer.optimize()
        self._save_fitness(
            results["best_so_far_x"], results["best_so_far_y"]
        )  # fitness evaluation
        return itemgetter("x", "y")(optimizer.get_data())

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        results["actor_losses"] = self.actor_losses
        results["critic_losses"] = self.critic_losses
        return results, (
            self.actor.state_dict(),
            self.critic.state_dict(),
            self.actor_optimizer.state_dict(),
            self.critic_optimizer.state_dict(),
        )

    def learn(self, advantage, log_prob):
        actor_loss = self.actor_loss_fn(advantage.detach(), log_prob)
        critic_loss = self.critic_loss_fn(advantage, torch.zeros_like(advantage))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        self.actor_losses.append(actor_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        self.critic_losses.append(critic_loss.item())

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        old_state = None
        x, y, reward = None, None, None
        while not self._check_terminations():
            state = self.get_state(x, y)
            with torch.no_grad():
                future_policy = self.actor(state.to(device))
                future_value = self.critic(state.to(device))
            if reward is not None and self.train_mode:
                current_value = DISCOUNT_FACTOR * future_value.detach() + reward
                predicted_value = self.critic(old_state.to(device))
                predicted_policy = self.actor(old_state.to(device))
                log_prob = torch.log(predicted_policy[self.last_choice])
                advantage = current_value - predicted_value
                self.learn(advantage, log_prob)

            probabilities = future_policy.cpu().detach().numpy()

            probabilities /= probabilities.sum()
            self.last_choice = np.random.choice([0, 1, 2], p=probabilities)
            action_options = {k: v for k, v in self.options.items()}
            action_options["max_function_evaluations"] = min(
                self.n_function_evaluations + self.sub_optimizer_max_fe,
                self.max_function_evaluations,
            )
            action_options["verbose"] = False
            optimizer = self.ACTIONS[self.last_choice](self.problem, action_options)
            optimizer.n_function_evaluations = self.n_function_evaluations
            optimizer._n_generations = 0
            best_parent = np.min(y) if y is not None else float("inf")
            x, y = self.iterate(x, y, optimizer)
            reward = get_reward(y, best_parent)
            self._print_verbose_info(fitness, y)
            if optimizer.best_so_far_y >= self.best_so_far_y:
                self.stagnation_count += (
                    optimizer.n_function_evaluations - self.n_function_evaluations
                )
            else:
                self.stagnation_count = 0
            self.n_function_evaluations = optimizer.n_function_evaluations
            old_state = state.clone()
        return self._collect(fitness, self.best_so_far_y)


def get_reward(y, best_parent):
    return max(best_parent - np.min(y), 0)
