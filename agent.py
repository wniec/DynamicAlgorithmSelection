import time

import numpy as np
import torch

from optimizers.G3PCX import G3PCX
from optimizers.LMCMAES import LMCMAES
from optimizers.Optimizer import Optimizer
from optimizers.SPSO import SPSO
from torch import nn

DISCOUNT_FACTOR = 0.9
device = "cuda" if torch.cuda.is_available() else ""


class ActorCritic(nn.Module):
    def __init__(self, state_dim=17, hidden_dim=64, action_dim=3):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # Separate heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor_head(features)
        value = self.critic_head(features)
        return action_probs, value


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, advantage, log_prob):
        return -advantage * log_prob


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
        self.model = ActorCritic().to(device)
        self.actor_loss_fn = ActorLoss().to(device)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.train_mode = options.get("train_mode", True)
        if p := options.get("model_parameters"):
            self.model.load_state_dict(p)
        if p := options.get("optimizer"):
            self.optimizer.load_state_dict(p)
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
        measured_individuals = [self.n_individuals // i for i in (2, 3, 4, 6, 9)]
        vector = [
            (np.argmin(y) - self.best_so_far_y)
            / ((self.worst_so_far_y - self.best_so_far_y) or 1),
            (average_y - self.best_so_far_y)
            / ((self.worst_so_far_y - self.best_so_far_y) or 1),
            sum((i - average_y) ** 2 for i in self.history)
            / (self.best_so_far_y - mid_so_far) ** 2
            / len(self.history),
            (self.max_function_evaluations - self.n_function_evaluations) / self.max_function_evaluations,
            self.ndim_problem / 40,  # maximum dimensionality in this COCO benchmark is 40
            self.stagnation_count / self.max_function_evaluations,
            *(
                dist(x[i], self.best_so_far_x) / dist_max
                for i in measured_individuals
            ),
            (dist(x[np.argmin(y)], self.best_so_far_x) / dist_max),
            *(
                (y[i] - np.min(y))
                / ((self.worst_so_far_y - self.best_so_far_y) or 1)
                for i in measured_individuals
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

    def _save_fitness(self, best_x, best_y, worst_x, worst_y):
        # update best-so-far solution (x) and fitness (y)
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
            results["best_so_far_x"], results["best_so_far_y"], results["worst_so_far_x"], results["worst_so_far_y"]
        )  # fitness evaluation
        return optimizer.get_data()

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        results["actor_losses"] = self.actor_losses
        results["critic_losses"] = self.critic_losses
        return results, (
            self.model.state_dict(),
            self.optimizer.state_dict(),
        )

    def learn(self, advantage, log_prob):
        advantage = torch.clamp(advantage, -5.0, 5.0)
        actor_loss = self.actor_loss_fn(advantage.detach(), log_prob)
        critic_loss = self.critic_loss_fn(advantage, torch.zeros_like(advantage))

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization
        old_state = None

        x, y, reward = None, None, None
        iteration_result = {'x': x, "y": y}
        while not self._check_terminations():
            state = self.get_state(x, y)
            with torch.no_grad():
                future_policy, future_value = self.model(state.to(device))
            if reward is not None and self.train_mode:
                current_value = DISCOUNT_FACTOR * future_value.detach() + reward
                predicted_policy, predicted_value = self.model(old_state.to(device))
                log_prob = torch.log(predicted_policy[self.last_choice])
                advantage = current_value - predicted_value
                self.learn(advantage, log_prob)
            probabilities = future_policy.cpu().detach().numpy()
            probabilities /= probabilities.sum()
            self.last_choice = np.random.choice([0, 1, 2], p=probabilities) if self.train_mode else np.argmax(
                probabilities)
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
            iteration_result = self.iterate(iteration_result, optimizer)
            x, y = iteration_result.get("x"), iteration_result.get("y")
            reward = self.get_reward(y, best_parent)
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

    def get_reward(self, y, best_parent):
        # reference = min(self.worst_so_far_y, best_parent)
        improvement = (best_parent - np.min(y))
        reward = np.sign(improvement)  # 1 for improvement, 0 for no change, and -1 for worsening
        return reward
