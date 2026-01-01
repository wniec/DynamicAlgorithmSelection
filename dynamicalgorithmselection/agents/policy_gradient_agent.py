import os

import numpy as np
import torch

from dynamicalgorithmselection.agents.agent_state import get_padded_population_observation
from dynamicalgorithmselection.agents.agent_utils import (
    RolloutBuffer,
    DEVICE,
    compute_gae,
    ActorLoss,
    ActorCritic,
)
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class PolicyGradientAgent(Agent):
    def __init__(self, problem, options):
        Agent.__init__(self, problem, options)
        self.buffer = options.get("buffer") or RolloutBuffer(
            capacity=options.get("ppo_batch_size", 5_000), device=DEVICE
        )
        self.model = ActorCritic(n_actions=len(self.actions)).to(DEVICE)
        self.actor_loss_fn = ActorLoss().to(DEVICE)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        decay_gamma = self.options.get("lr_decay_gamma", 0.9995)
        if p := options.get("model_parameters", None):
            self.model.load_state_dict(p)
        if p := options.get("optimizer", None):
            self.optimizer.load_state_dict(p)

        self.mean_rewards = options.get("mean_rewards", [])
        self.best_50_mean = float("inf")

        self.tau = self.options.get("critic_target_tau", 0.05)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, gamma=decay_gamma, step_size=10
        )

    def ppo_update(
        self,
        buffer,
        epochs=4,
        minibatch_size=512,
        clip_eps=0.3,
        value_coef=0.3,
        entropy_coef=0.02,
    ):
        states, actions, old_log_probs, values, rewards, dones, populations = buffer.as_tensors()
        with torch.no_grad():
            last_value = (
                self.model(states[-1].unsqueeze(0).to(DEVICE), populations.to(DEVICE)).squeeze(0).cpu().item()
                if buffer.size() > 0
                else 0.0
            )[1]
        returns, advantages = compute_gae(
            rewards,
            dones,
            values.detach().cpu(),
            last_value,
        )
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = advantages.to(DEVICE)
        returns = returns.to(DEVICE)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = states.shape[0]

        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                # clipping mb_idx so it doesn't cover next episode
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_populations = populations[mb_idx]
                # forward
                policy = self.model(mb_states, mb_populations)[0]
                dist_log_probs = torch.log(
                    policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-12
                )
                entropy = -(policy * torch.log(policy + 1e-12)).sum(dim=1).mean()
                # ratio and clipped surrogate
                ratio = torch.exp(dist_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                # value loss
                values_pred = self.model(mb_states, mb_populations)[0].squeeze(1)
                value_loss = torch.nn.functional.mse_loss(values_pred, mb_returns)
                loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
                # optimize
                self.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

            if self.run:
                self.run.log(
                    {
                        "actor_loss": actor_loss.detach().item(),
                        "critic_loss": value_loss.detach().item(),
                    }
                )

    def _collect(self, fitness, y=None):
        results, _ = super()._collect(fitness, y)
        self.mean_rewards.append(sum(self.rewards) / len(self.rewards))
        agent_state = {
            "model_parameters": self.model.state_dict(),
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

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)

        batch_size = self.buffer.capacity
        ppo_epochs = self.options.get("ppo_epochs", 6)
        clip_eps = self.options.get("ppo_eps", 0.3)
        entropy_coef = 0.0
        value_coef = self.options.get("ppo_value_coef", 0.03)

        x, y, reward = None, None, None
        iteration_result = {"x": x, "y": y}
        idx = 0
        while not self._check_terminations():
            state_vector = self.get_state(x, y).numpy()  # Get raw values
            normalized_state = self.state_normalizer.normalize(
                state_vector.reshape(1, -1)
            )  # Normalize
            state = torch.tensor(normalized_state, dtype=torch.float)  # Back to tensor
            state = torch.nan_to_num(state, nan=0.5, neginf=0.0, posinf=1.0)
            population_vector = get_padded_population_observation(x, y, pop_size=self.n_individuals)

            with torch.no_grad():
                policy, value = self.model(state.to(DEVICE), population_vector.to(DEVICE))

            probs = (
                policy.cpu().numpy().squeeze(0)
                if self.buffer.size() >= self.buffer.capacity
                else np.ones_like(self.actions, dtype=float) / len(self.actions)
            )
            probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
            probs /= probs.sum()
            if self.run is not None:
                self.run.log(
                    {
                        "normalized entropy": np.sum(-probs * np.log(probs))
                        / np.log(len(probs))
                    }
                )

            action = np.random.choice(len(probs), p=probs)
            self.choices_history.append(action)
            log_prob = torch.log(policy[0, action] + 1e-12).detach()
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
            reward = self.reward_normalizer.normalize(reward, idx)
            self.rewards.append(reward)
            if self.run:
                self.run.log({"reward": reward})
            self.n_function_evaluations = optimizer.n_function_evaluations
            self.buffer.add(
                state.squeeze(0).to(DEVICE),
                action,
                float(reward),
                self.n_function_evaluations == self.max_function_evaluations,
                log_prob,
                value.detach(),
                population_vector
            )

            # every batch_size steps or on termination, run ppo update
            if self.train_mode and self.buffer.size() >= batch_size:
                self.ppo_update(
                    self.buffer,
                    epochs=ppo_epochs,
                    clip_eps=clip_eps,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )
            entropy_coef = max(entropy_coef * 0.99997, 1e-2)
            self._print_verbose_info(fitness, y)
            if optimizer.best_so_far_y >= self.best_so_far_y:
                self.stagnation_count += (
                    optimizer.n_function_evaluations - self.n_function_evaluations
                )
            else:
                self.stagnation_count = 0

            self.n_function_evaluations = optimizer.n_function_evaluations
            idx += 1
        return self._collect(fitness, self.best_so_far_y)
