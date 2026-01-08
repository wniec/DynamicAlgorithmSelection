import torch
from torch import nn
import torch.nn.init as init


GAMMA = 0.3
HIDDEN_SIZE = 144
BASE_STATE_SIZE = 59
LAMBDA = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, n_actions: int, pop_size: int, dropout_p: float = 0.35):
        super().__init__()
        # Replace LSTM with a standard MLP feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(pop_size * 16, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Optional: Added dropout to feature extraction
        )

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Softmax(dim=-1),
        )
        orthogonal_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input shapes that might come with a sequence dimension (B, Seq, Feat)
        # or just (B, Feat). We flatten sequence dim if present because this is now an MLP.
        if x.dim() == 3:
            x = x.squeeze(1)

        x = x.to(next(self.parameters()).device)
        features = self.feature_extractor(x)
        return self.head(features)


class Critic(nn.Module):
    def __init__(self, pop_size: int, dropout_p: float = 0.35):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(pop_size * 16, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        orthogonal_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)

        x = x.to(next(self.parameters()).device)
        features = self.feature_extractor(x)
        return self.head(features)


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, advantage, log_prob):
        return -advantage * log_prob


def orthogonal_init(module):
    for name, param in module.named_parameters():
        if "weight_ih" in name:  # LSTM input -> hidden
            init.orthogonal_(param.data)
        elif "weight_hh" in name:  # LSTM hidden -> hidden
            init.orthogonal_(param.data)
        elif "weight" in name and param.dim() >= 2:
            init.orthogonal_(param.data)  # Linear layers
        elif "bias" in name:
            param.data.zero_()
