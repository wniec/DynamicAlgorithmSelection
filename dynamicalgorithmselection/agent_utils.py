import numpy as np
import torch
from torch import nn


DISCOUNT_FACTOR = 0.9
HIDDEN_SIZE = 192
BASE_STATE_SIZE = 60
ALPHA = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else ""


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
    # rewards_arr = np.array(rewards)
    # rewards = (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-5)
    T = len(rewards)
    returns = torch.zeros(T, device=DEVICE)
    advantages = torch.zeros(T, device=DEVICE)
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
    def __init__(self, n_actions: int, dropout_p: float = 0.15):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(BASE_STATE_SIZE + n_actions, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.actor(x)


class Critic(nn.Module):
    def __init__(self, n_actions: int):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(BASE_STATE_SIZE + n_actions, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=0.1),
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
