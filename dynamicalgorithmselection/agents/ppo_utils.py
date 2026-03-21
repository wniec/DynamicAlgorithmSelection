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


def compute_gae(rewards, dones, values):
    T = len(rewards)
    returns = torch.zeros(T, device=DEVICE)
    advantages = torch.zeros(T, device=DEVICE)
    prev_return = 0.0
    prev_value = 0.0
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
    def __init__(self, d_dim, num_algorithms):
        super(RLDASNetwork, self).__init__()
        self.L = num_algorithms
        self.D = d_dim

        self.actor = RLDASActor(d_dim, num_algorithms, DEVICE)
        self.critic = RLDASCritic(d_dim, num_algorithms, DEVICE)

    def forward(self, la_state, ah_state):
        return self.actor(la_state, ah_state), self.critic(la_state, ah_state)


class RLDASActor(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.optimizer_num = optimizer_num
        self.embedders = nn.ModuleList(
            [
                (
                    nn.Sequential(
                        nn.Linear(dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.ReLU(),
                    )
                ).to(device)
                for _ in range(2 * optimizer_num)
            ]
        )

        self.embedder_final = nn.Sequential(
            nn.Linear(9 + optimizer_num * 2, 64),
            nn.Tanh(),
        ).to(device)
        self.model = nn.Sequential(
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, optimizer_num),
            nn.Softmax(dim=-1),
        ).to(device)

    def forward(self, la_state, ah_state):
        flattened_ah_state = torch.flatten(ah_state, start_dim=1, end_dim=2)

        embedded_ah = [
            embedder(flattened_ah_state[:, i, :])
            for i, embedder in enumerate(self.embedders)
        ]

        embedded_ah = torch.cat(embedded_ah, dim=-1)
        batch_size = ah_state.shape[0]
        x = torch.cat((la_state, embedded_ah), dim=-1).view(batch_size, -1)
        x = self.embedder_final(x)
        probs = self.model(x)

        return probs


class RLDASCritic(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList(
            [
                (
                    nn.Sequential(
                        nn.Linear(dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.ReLU(),
                    )
                ).to(device)
                for _ in range(2 * optimizer_num)
            ]
        )
        self.embedder_final = nn.Sequential(
            nn.Linear(9 + optimizer_num * 2, 64),
            nn.Tanh(),
        ).to(device)
        self.model = nn.Sequential(
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ).to(device)

    def forward(self, la_state, ah_state):
        flattened_ah_state = torch.flatten(ah_state, start_dim=1, end_dim=2)
        embedded_ah = [
            embedder(flattened_ah_state[:, i, :])
            for i, embedder in enumerate(self.embedders)
        ]
        embedded_ah = torch.cat(embedded_ah, dim=-1)
        batch_size = ah_state.shape[0]
        feature = torch.cat((la_state, embedded_ah), dim=-1).view(batch_size, -1)
        feature = self.embedder_final(feature)
        val = self.model(feature.view(batch_size, -1))

        return val
