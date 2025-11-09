import numpy as np
import torch

from dynamicalgorithmselection.agents.agent_utils import (
    RolloutBuffer,
    DEVICE,
    compute_gae,
    DISCOUNT_FACTOR,
    Actor,
    Critic,
    ActorLoss,
)
from dynamicalgorithmselection.agents.agent import Agent
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


class PolicyGradientAgent(Agent):
    def __init__(self, problem, options):
        Agent.__init__(self, problem, options)
        self.buffer = RolloutBuffer(
            capacity=options.get("ppo_batch_size", 1024), device=DEVICE
        )
        self.actor = Actor(n_actions=len(self.actions)).to(DEVICE)
        self.critic = Critic(n_actions=len(self.actions)).to(DEVICE)
        self.actor_loss_fn = ActorLoss().to(DEVICE)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-6)
        decay_gamma = self.options.get("lr_decay_gamma", 0.9998)
        if p := options.get("actor_parameters", None):
            self.actor.load_state_dict(p)
        if p := options.get("critic_parameters", None):
            self.critic.load_state_dict(p)
        if p := options.get("actor_optimizer", None):
            self.actor_optimizer.load_state_dict(p)
        if p := options.get("critic_optimizer", None):
            self.critic_optimizer.load_state_dict(p)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=decay_gamma
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=decay_gamma
        )

    def ppo_update(
        self, buffer, epochs=2, clip_eps=0.2, value_coef=0.05, entropy_coef=0.05
    ):
        states, actions, old_log_probs, values, rewards, dones = buffer.as_tensors()

        with torch.no_grad():
            last_value = (
                self.critic(states[-1].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().item()
            )
        returns, advantages = compute_gae(
            rewards,
            dones,
            values.detach().cpu(),
            last_value,
            gamma=DISCOUNT_FACTOR,
            lam=0.95,
        )
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
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
        return results, {
            "actor_parameters": self.actor.state_dict(),
            "critic_parameters": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def optimize(self, fitness_function=None, args=None):
        fitness = Optimizer.optimize(self, fitness_function)

        batch_size = self.options.get("sub_optimization_ratio", 10)
        ppo_epochs = self.options.get("ppo_epochs", 10)
        clip_eps = self.options.get("ppo_eps", 0.2)
        entropy_coef = self.options.get("ppo_entropy", 0.05)
        value_coef = self.options.get("ppo_value_coef", 0.3)

        x, y, reward = None, None, None
        iteration_result = {"x": x, "y": y}
        while not self._check_terminations():
            state = self.get_state(x, y).unsqueeze(0)
            state = torch.nan_to_num(state, nan=0.5, neginf=0.0, posinf=1.0)
            with torch.no_grad():
                policy = self.actor(state.to(DEVICE))
                value = self.critic(state.to(DEVICE))

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
                state.squeeze(0).to(DEVICE),
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
            entropy_coef = max(entropy_coef * 0.999, 0.01)
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
