import torch
from torch import nn
import torch.nn.init as init
import numpy as np

GAMMA = 0.8
HIDDEN_SIZE = 144
LAMBDA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes a layer.
    - If std is 0, we force weights to be exactly 0 (Hard Zero Init).
    - Otherwise, we use Orthogonal Init with the given gain/std.
    """
    if std == 0.0:
        init.constant_(layer.weight, 0.0)
    else:
        init.orthogonal_(layer.weight, std)

    init.constant_(layer.bias, bias_const)
    return layer


class RolloutBuffer:
    def __init__(self, capacity, device=DEVICE):
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

        states = torch.stack(self.states)[-self.capacity :].to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)[
            -self.capacity :
        ]
        old_log_probs = torch.stack(self.log_probs).to(self.device)[-self.capacity :]
        values = torch.stack(self.values).squeeze(-1).to(self.device)[-self.capacity :]
        rewards = self.rewards[-self.capacity :]
        dones = self.dones[-self.capacity :]
        return states, actions, old_log_probs, values, rewards, dones


def compute_gae(rewards, dones, values, last_value):
    T = len(rewards)
    returns = torch.zeros(T, device=DEVICE)
    advantages = torch.zeros(T, device=DEVICE)
    prev_return = last_value
    prev_value = last_value
    prev_adv = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + GAMMA * prev_value * mask - values[t]
        adv = delta + GAMMA * LAMBDA * prev_adv * mask
        advantages[t] = adv
        prev_adv = adv
        prev_value = values[t]
        prev_return = rewards[t] + GAMMA * prev_return * mask
        returns[t] = prev_return
    return returns, advantages


class Actor(nn.Module):
    def __init__(self, n_actions: int, input_size: int, dropout_p: float = 0.35):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(input_size, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.head = nn.Sequential(
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            layer_init(nn.Linear(HIDDEN_SIZE, n_actions), std=0.0),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        features = self.feature_extractor(x)
        return self.head(features)


class Critic(nn.Module):
    def __init__(self, input_size: int, dropout_p: float = 0.35):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(input_size, HIDDEN_SIZE)),  # std=sqrt(2)
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.head = nn.Sequential(
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        features = self.feature_extractor(x)
        return self.head(features)


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, advantage, log_prob):
        return -advantage * log_prob


class RLDASNetwork(nn.Module):
    def __init__(self, d_dim, num_algorithms, la_dim=9):
        super(RLDASNetwork, self).__init__()
        self.L = num_algorithms
        self.D = d_dim
        self.la_dim = la_dim

        self.ah_input_flat_dim = self.L * 2 * self.D

        self.ah_embed = nn.Sequential(
            nn.Linear(self.ah_input_flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.L),  # Output size aligned with paper description
            nn.ReLU(),
        )
        self.fusion_input_dim = self.la_dim + (2 * self.L)

        self.dv_layer = nn.Sequential(nn.Linear(self.fusion_input_dim, 64), nn.Tanh())

        self.actor_head = nn.Sequential(
            nn.Linear(64, 16), nn.Tanh(), nn.Linear(16, self.L), nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Scalar Value
        )

    def forward(self, la_state, ah_state):
        if ah_state.dim() > 2:
            batch_size = ah_state.size(0)
            ah_flat = ah_state.view(batch_size, -1)
        else:
            ah_flat = ah_state

        v_ah = self.ah_embed(ah_flat)

        combined = torch.cat([la_state, v_ah], dim=1)

        dv = self.dv_layer(combined)

        probs = self.actor_head(dv)
        value = self.critic_head(dv)

        return probs, value
