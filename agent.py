import time
from operator import itemgetter

import numpy as np
import torch

from optimizers.Optimizer import Optimizer
from torch import nn

DISCOUNT_FACTOR = 0.9
HIDDEN_SIZE = 128
STATE_SIZE = 63
ALPHA = 0.3
device = "cuda" if torch.cuda.is_available() else ""


class RolloutBuffer:
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def size(self):
        return len(self.states)

    def as_tensors(self):
        import torch

        states = torch.stack(self.states)[-self.capacity :].to(
            self.device
        )  # shape (T, state_dim)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)[
            -self.capacity :
        ]  # (T,)
        old_log_probs = torch.stack(self.log_probs).to(self.device)[
            -self.capacity :
        ]  # (T,)
        values = (
            torch.stack(self.values).squeeze(-1).to(self.device)[-self.capacity :]
        )  # (T,)
        rewards = self.rewards[-self.capacity :]
        dones = self.dones[-self.capacity :]
        return states, actions, old_log_probs, values, rewards, dones


def compute_gae(rewards, dones, values, last_value, gamma=0.85, lam=0.85):
    # rewards: list of scalars length T
    # dones: list of bools length T
    # values: tensor shape (T,)    (values for states 0..T-1)
    # last_value: scalar (value estimate for final next state)
    import torch

    T = len(rewards)
    returns = torch.zeros(T, device=device)
    advantages = torch.zeros(T, device=device)
    prev_return = last_value
    prev_value = last_value
    prev_adv = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * prev_value * mask - values[t]
        adv = delta + gamma * lam * prev_adv * mask
        advantages[t] = adv
        prev_adv = adv
        prev_value = values[t]
        prev_return = rewards[t] + gamma * prev_return * mask
        returns[t] = prev_return
    return returns, advantages


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.actor(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(STATE_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.critic(x)


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, advantage, log_prob):
        return -advantage * log_prob


def get_weighted_central_moment(n: int, weights, norms_squared):
    exponent = n / 2
    numerator = min((weights * norms_squared**exponent).sum(), 1e8)
    inertia_denom_w = np.linalg.norm(weights)
    inertia_denom_n = np.linalg.norm(norms_squared**exponent)
    return numerator / max(1e-5, inertia_denom_w * max(1e-5, inertia_denom_n))


class Agent(Optimizer):
    def __init__(self, problem, options):
        Optimizer.__init__(self, problem, options)
        self.rewards = []
        self.buffer = RolloutBuffer(
            capacity=options.get("ppo_batch_size", 1024), device=device
        )
        self.choices_history = []
        self.stagnation_count = 0
        self._n_generations = 0
        self.problem = problem
        self.rewards = []
        self.options = options
        self.history = []
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.actor_loss_fn = ActorLoss().to(device)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=8e-6)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=5e-7)
        decay_gamma = self.options.get("lr_decay_gamma", 0.9998)
        self.train_mode = options.get("train_mode", True)
        if p := options.get("actor_parameters"):
            self.actor.load_state_dict(p)
        if p := options.get("critic_parameters"):
            self.critic.load_state_dict(p)
        if p := options.get("actor_optimizer"):
            self.actor_optimizer.load_state_dict(p)
        if p := options.get("critic_optimizer"):
            self.critic_optimizer.load_state_dict(p)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=decay_gamma
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=decay_gamma
        )
        sub_optimization_ratio = options["sub_optimization_ratio"]
        self.run = options.get("run", None)
        self.sub_optimizer_max_fe = (
            self.max_function_evaluations / sub_optimization_ratio
        )

        self.actions = options.get("action_space")

    def get_state(self, x: np.ndarray, y: np.ndarray) -> torch.Tensor:
        # Definition of state is inspired by its representation in DE-DDQN article
        last_action_index = (
            self.choices_history[-1] if self.choices_history else len(self.actions)
        )
        last_action_encoded = [
            float(i) for i in np.binary_repr(last_action_index, width=4)[-2:]
        ]
        same_action_counter = 0
        choices_count = [
            sum(1 for i in self.choices_history if i == j)
            / (len(self.choices_history) or 1)
            for j in range(len(self.actions))
        ]
        for i in reversed(self.choices_history):
            if i == last_action_index:
                same_action_counter += 1
            else:
                break
        if x is None or y is None:
            vector = [
                0.0,  # third weighted central moment
                0.0,  # second weighted central moment
                0.0,  # normalized domination of best solution
                0.0,  # normalized radius of the smallest sphere containing entire population
                0.5,  # normalized relative fitness difference
                0.5,  # average_y relative to best
                1.0,  # variance measure
                1.0,  # full remaining budget (max evaluations)
                self.ndim_problem / 40,  # normalized problem dimension
                0.0,  # stagnation count
                *([0.0] * 29),
                *last_action_encoded,
                *([0.0] * 22),
            ]
        else:
            dist = lambda x1, x2: np.linalg.norm(x1 - x2)
            log_scaling = lambda x: x / (x + 5)
            dist_max = dist(self.lower_boundary, self.upper_boundary)
            x_std = 2 * x.std(axis=0) / (self.upper_boundary - self.lower_boundary)
            norm_x = (x - x.mean(axis=0)) / (x.std() + 1e-8)
            norm_dist = lambda x1, x2: min((dist(x1, x2) / dist_max), 1)
            average_y = sum(self.history) / (len(self.history) or 1)
            average_x = x.mean(axis=0)
            population_relative = x - average_x
            population_radius = np.linalg.norm(
                population_relative.max(axis=0) - population_relative.min(axis=0)
            )
            lower_x = x.min(axis=0)
            upper_x = x.max(axis=0)
            mid_so_far = (self.worst_so_far_y - self.best_so_far_y) / 2
            weights = (
                (1.0 - (y - y.min()) / (y.max() - y.min()))
                if (y.max() - y.min() > 1e-6)
                else np.ones_like(y)
            )
            weights_normalized = weights / weights.sum()
            norms = np.linalg.norm(population_relative, ord=2, axis=1)  # shape (pop,)
            second_weighted_central_moment = get_weighted_central_moment(
                2, weights_normalized, norms
            )
            third_weighted_central_moment = get_weighted_central_moment(
                3, weights_normalized, norms
            )
            i_sorted = sorted([i for i, _ in enumerate(y)], key=lambda i: y[i])
            y_normalized = (y - y.mean()) / (y.std() + 1e-6)
            y_diffs = [
                min(y_normalized[j] - y_normalized[i], 1)
                for i, j in zip(i_sorted, i_sorted[1:])
            ]
            slopes = [
                log_scaling(
                    max(y_normalized[j] - y_normalized[i], 1)
                    / (norm_dist(x[i], x[j]) + 1e-6)
                )
                for i, j in zip(i_sorted, i_sorted[1:])
            ]
            measured_individuals = list(
                itemgetter(1, 2, 3, 4, 5, 6, 9, 12, 15)(i_sorted)
            )
            choices_entropy = -(
                np.array(choices_count)
                * np.nan_to_num(np.log(choices_count), neginf=0, posinf=0, nan=0)
            ).sum() / np.log(len(choices_count))
            vector = [
                third_weighted_central_moment,
                second_weighted_central_moment,
                (y - self.best_so_far_y).mean()
                / (
                    max((y.max() - self.best_so_far_y), (y - self.best_so_far_y).sum())
                    or 1
                ),
                population_radius / dist_max,
                max(0.0, (np.min(y) - self.best_so_far_y))
                / ((self.worst_so_far_y - self.best_so_far_y) or 1),
                (average_y - self.best_so_far_y)
                / ((self.worst_so_far_y - self.best_so_far_y) or 1),
                sum(
                    (i - average_y) ** 2 for i in self.history
                )  # wcześniejsze najlepsze y
                / (self.best_so_far_y - mid_so_far)
                / (self.best_so_far_y - mid_so_far)
                / len(self.history),
                (self.max_function_evaluations - self.n_function_evaluations)
                / self.max_function_evaluations,
                self.ndim_problem
                / 40,  # maximum dimensionality in this COCO benchmark is 40
                self.stagnation_count / self.max_function_evaluations,
                *(norm_dist(x[i], self.best_so_far_x) for i in measured_individuals),
                (norm_dist(x[np.argmin(y)], self.best_so_far_x)),
                *(
                    norm_dist(x[i], average_x)
                    for i in measured_individuals + [np.argmin(y)]
                ),
                *(
                    (y[i] - np.min(y))
                    / max((self.worst_so_far_y - self.best_so_far_y), 1e-6)
                    for i in measured_individuals
                ),
                *last_action_encoded,
                same_action_counter
                / (self.max_function_evaluations / self.sub_optimizer_max_fe),
                *choices_count,
                (
                    np.prod(
                        (upper_x - lower_x)
                        / (self.upper_boundary - self.lower_boundary)
                    )
                )
                ** (1 / self.ndim_problem),  # searched volume
                x_std.max(),
                x_std.min(),
                x_std.mean(),
                2 * x_std.std(),
                np.clip((norm_x**2).mean(), -1, 1),
                np.clip((norm_x**2).min(), -1, 1),
                np.clip((norm_x**2).max(), -1, 1),
                np.clip((norm_x**2).std(), -1, 1),
                choices_entropy,
                norm_dist(self.best_so_far_x, self.worst_so_far_x),
                norm_dist(x[np.argmin(y)], x[np.argmax(y)]),
                max(y_diffs),
                min(y_diffs),
                sum(y_diffs) / len(y_diffs),
                max(slopes),
                min(slopes),
                sum(slopes) / len(slopes),
            ]
        return torch.tensor(vector, dtype=torch.float)

    def ppo_update(
        self, buffer, epochs=2, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01
    ):
        states, actions, old_log_probs, values, rewards, dones = buffer.as_tensors()

        with torch.no_grad():
            last_value = (
                self.critic(states[-1].unsqueeze(0).to(device)).squeeze(0).cpu().item()
            )
        returns, advantages = compute_gae(
            rewards,
            dones,
            values.detach().cpu(),
            last_value,
            gamma=DISCOUNT_FACTOR,
            lam=0.95,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -5, 5)

        for epoch in range(epochs):
            policy = self.actor(states)
            dist_log_probs = torch.log(
                policy.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-12
            )
            entropy = -(policy * torch.log(policy + 1e-12)).sum(dim=1).mean()

            ratio = torch.exp(dist_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            values_pred = self.critic(states).squeeze(1)
            value_loss = torch.nn.functional.mse_loss(values_pred, returns)
            loss = actor_loss + value_coef * value_loss
            if self.run:
                self.run.log(
                    {
                        "actor_loss": actor_loss.detach().item(),
                        "critic_loss": value_loss.detach().item(),
                    }
                )
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

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
            results["best_so_far_x"],
            results["best_so_far_y"],
            results["worst_so_far_x"],
            results["worst_so_far_y"],
        )  # fitness evaluation
        return optimizer.get_data()

    def _collect(self, fitness, y=None):
        if y is not None:
            self._print_verbose_info(fitness, y)
        results = Optimizer._collect(self, fitness)
        results["_n_generations"] = self._n_generations
        return results, {
            "actor_parameters": self.actor.state_dict(),
            "critic_parameters": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def optimize(self, fitness_function=None, args=None):
        self.start_time = time.time()
        if fitness_function is not None:
            self.fitness_function = fitness_function
        fitness = []  # to store all fitness generated during evolution/optimization

        batch_size = self.options.get("sub_optimization_ratio", 10)
        ppo_epochs = self.options.get("ppo_epochs", 10)
        clip_eps = self.options.get("ppo_eps", 0.2)
        entropy_coef = self.options.get("ppo_entropy", 0.01)
        value_coef = self.options.get("ppo_value_coef", 0.5)

        x, y, reward = None, None, None
        iteration_result = {"x": x, "y": y}
        while not self._check_terminations():
            state = self.get_state(x, y).unsqueeze(0)
            state = torch.nan_to_num(state, nan=0.5, neginf=0.0, posinf=1.0)
            with torch.no_grad():
                policy = self.actor(state.to(device))
                value = self.critic(state.to(device))

            probs = policy.cpu().numpy().squeeze(0)
            probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
            probs /= probs.sum()

            action = np.random.choice(len(probs), p=probs)
            self.choices_history.append(action)
            log_prob = torch.log(policy[0, action] + 1e-12).detach()
            action_options = {k: v for k, v in self.options.items()}
            action_options["max_function_evaluations"] = min(
                self.n_function_evaluations + self.sub_optimizer_max_fe,
                self.max_function_evaluations,
            )
            action_options["verbose"] = False
            optimizer = self.actions[action](self.problem, action_options)
            optimizer.n_function_evaluations = self.n_function_evaluations
            optimizer._n_generations = 0
            best_parent = np.min(y) if y is not None else float("inf")
            iteration_result = self.iterate(iteration_result, optimizer)
            x, y = iteration_result.get("x"), iteration_result.get("y")

            reward = self.get_reward(y, best_parent)
            self.rewards.append(reward)
            if self.run:
                self.run.log({"reward": reward})
            self.buffer.add(
                state.squeeze(0).to(device),
                action,
                float(reward),
                False,
                log_prob,
                value.detach(),
            )

            self.n_function_evaluations = optimizer.n_function_evaluations
            # every batch_size steps or on termination, run ppo update
            if self.train_mode and self.buffer.size() >= batch_size:
                self.ppo_update(
                    self.buffer,
                    epochs=ppo_epochs,
                    clip_eps=clip_eps,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )
            entropy_coef = max(entropy_coef * 0.999, 0.0005)
            self._print_verbose_info(fitness, y)
            if optimizer.best_so_far_y >= self.best_so_far_y:
                self.stagnation_count += (
                    optimizer.n_function_evaluations - self.n_function_evaluations
                )
            else:
                self.stagnation_count = 0

            self.n_function_evaluations = optimizer.n_function_evaluations
        # self.buffer.clear()
        return self._collect(fitness, self.best_so_far_y)

    def get_reward(self, y, best_parent):
        improvement = best_parent - np.min(y)
        reference = self.worst_so_far_y - self.best_so_far_y
        used_fe = self.n_function_evaluations / self.max_function_evaluations
        """
        scaled = 100 * improvement / reference
        log_scaled = np.sign(scaled) * np.log(np.abs(scaled) + 1)
        reward = np.tanh(log_scaled)
        if len(self.choices_history) > 1:
            reward += 0.05 if self.choices_history[-1] == self.choices_history[-2] else 0.0
        else:
            return 0"""
        reward = np.sign(improvement) * used_fe
        return reward  # to the 1/dim power ?
