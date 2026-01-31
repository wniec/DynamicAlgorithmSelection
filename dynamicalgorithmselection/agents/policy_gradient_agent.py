import os
import numpy as np
import torch

from dynamicalgorithmselection.agents.ppo_utils import (
    RolloutBuffer,
    DEVICE,
    compute_gae,
    Actor,
    Critic,
    ActorLoss,
)
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class PolicyGradientAgent(Agent):
    def __init__(self, problem, options):
        Agent.__init__(self, problem, options)
        self.buffer = options.get("buffer") or RolloutBuffer(
            capacity=options.get("ppo_batch_size", 2_500), device=DEVICE
        )
        self.actor = Actor(n_actions=len(self.actions), input_size=self.state_dim).to(
            DEVICE
        )
        self.critic = Critic(input_size=self.state_dim).to(DEVICE)
        self.actor_loss_fn = ActorLoss().to(DEVICE)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-5)

        self._load_parameters(options)

        self.mean_rewards = options.get("mean_rewards", [])
        self.best_50_mean = float("inf")

        self.tau = self.options.get("critic_target_tau", 0.05)
        self.target_kl = 0.03

        # Initialize history dict
        self.iterations_history = {"x": None, "y": None}

    def _load_parameters(self, options):
        """Loads state dicts if provided in options."""
        if p := options.get("actor_parameters", None):
            self.actor.load_state_dict(p)
        if p := options.get("critic_parameters", None):
            self.critic.load_state_dict(p)
        if p := options.get("actor_optimizer", None):
            self.actor_optimizer.load_state_dict(p)
        if p := options.get("critic_optimizer", None):
            self.critic_optimizer.load_state_dict(p)

    def _update_learning_rate(self, mean_kl):
        current_lr = self.actor_optimizer.param_groups[0]["lr"]

        if mean_kl > self.target_kl * 1.5:
            current_lr /= 1.5
        elif mean_kl < self.target_kl / 1.5:
            current_lr *= 1.5

        current_lr = np.clip(current_lr, 3e-6, 1e-4)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = current_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def _compute_advantages(self, buffer):
        """Computes GAE returns and advantages."""
        states, actions, old_log_probs, values, rewards, dones = buffer.as_tensors()
        with torch.no_grad():
            last_value = (
                self.critic(states[-1].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().item()
                if buffer.size() > 0
                else 0.0
            )
        returns, advantages = compute_gae(
            rewards, dones, values.detach().cpu(), last_value
        )
        advantages = advantages.to(DEVICE)
        returns = returns.to(DEVICE)
        return states, actions, old_log_probs, returns, advantages

    def _update_on_minibatch(
        self,
        mb_states,
        mb_actions,
        mb_old_log_probs,
        mb_returns,
        mb_advantages,
        clip_eps,
        value_coef,
        entropy_coef,
    ):
        """Performs a single update step on a minibatch."""
        policy = self.actor(mb_states)
        dist_log_probs = torch.log(
            policy.gather(1, mb_actions.unsqueeze(1)).squeeze(1) + 1e-12
        )
        entropy = -(policy * torch.log(policy + 1e-12)).sum(dim=1).mean()

        ratio = torch.exp(dist_log_probs - mb_old_log_probs)

        with torch.no_grad():
            log_ratio = dist_log_probs - mb_old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()

        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        values_pred = self.critic(mb_states).squeeze(1)
        value_loss = torch.nn.functional.mse_loss(values_pred, mb_returns)

        loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.detach().item(), value_loss.detach().item(), approx_kl.item()

    def ppo_update(
        self,
        buffer,
        epochs=4,
        minibatch_size=256,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        states, actions, old_log_probs, returns, advantages = self._compute_advantages(
            buffer
        )
        dataset_size = states.shape[0]

        total_approx_kl = 0.0
        n_batches = 0
        last_actor_loss = 0.0
        last_value_loss = 0.0

        for epoch in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]

                a_loss, v_loss, kl = self._update_on_minibatch(
                    states[mb_idx],
                    actions[mb_idx],
                    old_log_probs[mb_idx],
                    returns[mb_idx],
                    advantages[mb_idx],
                    clip_eps,
                    value_coef,
                    entropy_coef,
                )

                last_actor_loss, last_value_loss = a_loss, v_loss
                total_approx_kl += kl
                n_batches += 1

        mean_kl = total_approx_kl / max(n_batches, 1)
        current_lr = self._update_learning_rate(mean_kl)

        if self.run:
            self.run.log(
                {
                    "actor_loss": last_actor_loss,
                    "critic_loss": last_value_loss,
                    "learning_rate": current_lr,
                    "approx_kl": mean_kl,
                }
            )

    def _collect(self, fitness, y=None):
        results, _ = super()._collect(fitness, y)
        self.mean_rewards.append(sum(self.rewards) / len(self.rewards))
        agent_state = {
            "actor_parameters": self.actor.state_dict(),
            "critic_parameters": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
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

    def _prepare_state_tensor(self, x, y, full_buffer):
        """Generates and normalizes the state tensor using self.iterations_history."""
        state = self.get_state(
            x,
            y,
            self.iterations_history["x"],
            self.iterations_history["y"],
            self.train_mode and not full_buffer,
        )
        state = torch.nan_to_num(
            torch.tensor(state), nan=0.5, neginf=0.0, posinf=1.0
        ).unsqueeze(0)
        return state.to(dtype=torch.float32)

    def _select_action(self, state, full_buffer):
        """Calculates policy, probabilities and selects an action."""
        with torch.no_grad():
            policy = self.actor(state.to(DEVICE))
            value = self.critic(state.to(DEVICE))

        if full_buffer:
            probs = policy.cpu().numpy().squeeze(0)
        else:
            probs = np.ones_like(self.actions, dtype=float) / len(self.actions)

        probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
        probs /= probs.sum()

        if self.run is not None:
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs))
            self.run.log({"normalized entropy": entropy})

        action = np.random.choice(len(probs), p=probs)
        log_prob = torch.log(policy[0, action] + 1e-12).detach()

        return action, log_prob, value

    def _execute_action(self, action_idx, iteration_result):
        """Instantiates and runs the selected optimizer."""
        action_options = {k: v for k, v in self.options.items()}
        action_options["max_function_evaluations"] = min(
            self.checkpoints[self._n_generations],
            self.max_function_evaluations,
        )
        action_options["verbose"] = False

        optimizer = self.actions[action_idx](self.problem, action_options)
        optimizer.n_function_evaluations = self.n_function_evaluations
        optimizer._n_generations = 0

        # Run optimization step
        return self.iterate(iteration_result, optimizer), optimizer

    def _update_history(self, iteration_result):
        """Merges new results into self.iterations_history"""
        for key, val in iteration_result.items():
            if key.endswith("_history") and key != "fitness_history":
                variable_name = key[: -len("_history")]
                appended_val = iteration_result.get(key)
                historic_val = self.iterations_history.get(variable_name)

                if historic_val is None:
                    self.iterations_history[variable_name] = appended_val
                else:
                    self.iterations_history[variable_name] = np.concatenate(
                        (historic_val, appended_val)
                    )

        return iteration_result

    def _process_step_reward(self, best_parent, idx, full_buffer):
        """Calculates and normalizes the reward."""
        new_best_y = self.best_so_far_y
        reward = self.get_reward(new_best_y, best_parent)

        reward = self.reward_normalizer.normalize(
            reward, idx, update=self.train_mode and not full_buffer
        )

        self.rewards.append(reward)
        if self.run:
            self.run.log({"reward": reward})

        return float(reward)

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)

        # PPO Hyperparameters
        batch_size = self.buffer.capacity
        ppo_epochs = self.options.get("ppo_epochs", 6)
        clip_eps = self.options.get("ppo_eps", 0.3)
        value_coef = self.options.get("ppo_value_coef", 0.3)
        entropy_coef = 0.01

        # State initialization
        x, y = None, None
        self.iterations_history = {"x": None, "y": None}
        iteration_result = {"x": x, "y": y}
        idx = 0
        last_used_params = []
        while not self._check_terminations():
            full_buffer = self.buffer.size() >= self.buffer.capacity

            # 1. Prepare State (uses self.iterations_history internally)
            state = self._prepare_state_tensor(x, y, full_buffer)

            # 2. Select Action
            action, log_prob, value = self._select_action(state, full_buffer)
            self.choices_history.append(action)

            # 3. Execute Optimization Step
            best_parent = self.best_so_far_y

            iteration_result, optimizer = self._execute_action(action, iteration_result)
            if len(last_used_params) > 0:
                for key in last_used_params:
                    if key in self.iterations_history and key not in iteration_result:
                        self.iterations_history.pop(key)
            last_used_params = optimizer.start_condition_parameters
            x, y = iteration_result.get("x"), iteration_result.get("y")

            # 4. Update and Deduplicate History (updates self.iterations_history internally)

            iteration_result = self._update_history(iteration_result)

            # 5. Process Reward
            reward = self._process_step_reward(best_parent, idx, full_buffer)

            # 6. Store in Buffer
            self.n_function_evaluations = optimizer.n_function_evaluations
            is_done = self.n_function_evaluations >= self.max_function_evaluations
            self.buffer.add(
                state.squeeze(0).to(DEVICE),
                action,
                reward,
                is_done,
                log_prob,
                value.detach(),
            )

            # 7. PPO Update if needed
            if self.train_mode and self.buffer.size() >= batch_size:
                self.ppo_update(
                    self.buffer,
                    epochs=ppo_epochs,
                    clip_eps=clip_eps,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                )

            # 8. Post-step updates
            entropy_coef = max(entropy_coef * 0.99, 0.001)
            self._print_verbose_info(fitness, y)

            # Stagnation Logic
            if optimizer.best_so_far_y >= self.best_so_far_y:
                self.stagnation_count += (
                    optimizer.n_function_evaluations - self.n_function_evaluations
                )
            else:
                self.stagnation_count = 0

            self.n_function_evaluations = optimizer.n_function_evaluations
            idx += 1

        return self._collect(fitness, self.best_so_far_y)
