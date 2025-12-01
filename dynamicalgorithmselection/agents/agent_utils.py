import numpy as np
import torch
from torch import nn

CHECKPOINT_DIVISION_EXPONENT = 1.8
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


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.hidden = None

    def reset_memory(self, batch_size=1, device=DEVICE):
        self.hidden = (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device),
        )

    def forward_lstm(self, x):
        batch_size = x.size(0)
        device = x.device

        # Reinitialize hidden state if needed
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.reset_memory(batch_size=batch_size, device=device)
        else:
            # Detach hidden state from previous graph
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        out, self.hidden = self.lstm(x, self.hidden)
        return out


class Actor(LSTMModule):
    def __init__(self, n_actions: int, dropout_p: float = 0.35, lstm_layers: int = 1):
        super().__init__(BASE_STATE_SIZE + n_actions * 2, HIDDEN_SIZE, lstm_layers)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
        elif x.dim() == 3:
            x_seq = x
        else:
            raise ValueError(f"Unsupported input tensor shape for Actor: {x.shape}")
        x_seq = x_seq.to(next(self.parameters()).device)
        lstm_out = self.forward_lstm(x_seq)
        last = lstm_out[:, -1, :]
        return self.head(last)


class Critic(LSTMModule):
    def __init__(self, n_actions: int, dropout_p: float = 0.35, lstm_layers: int = 1):
        super().__init__(BASE_STATE_SIZE + n_actions * 2, HIDDEN_SIZE, lstm_layers)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
        elif x.dim() == 3:
            x_seq = x
        else:
            raise ValueError(f"Unsupported input tensor shape for Critic: {x.shape}")
        x_seq = x_seq.to(next(self.parameters()).device)
        lstm_out = self.forward_lstm(x_seq)
        last = lstm_out[:, -1, :]
        return self.head(last)


class ActorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, advantage, log_prob):
        return -advantage * log_prob


def distance(x0: np.ndarray, x1: np.ndarray) -> float:
    return np.linalg.norm(x0 - x1)


def inverse_scaling(x):
    # Monotonic increacing in [0, inf) function that is bounded in [0, 1)
    return x / (x + 5)


def get_list_stats(data: list):
    return (
        max(data),
        min(data),
        sum(data) / len(data),
    )


def get_runtime_stats(
    fitness_history: list[tuple[int, float]],
    function_evaluations: int,
    checkpoints: list[int],
) -> dict[str, float | list[float]]:
    """
    :param fitness_history: list of tuples [fe, fitness] with only points where best so far fitness improved
    :param function_evaluations: max number of function evaluations during run.
    :param checkpoints: list of checkpoints by their n_function_evaluations
    :return: dictionary of selected run statistics, ready to dump
    """
    area_under_optimization_curve = 0.0
    last_i = 0
    checkpoint_idx = 0
    last_fitness = None
    checkpoints_fitness = []
    for i, fitness in fitness_history:
        area_under_optimization_curve += fitness * (i - last_i)
        while last_i <= checkpoints[checkpoint_idx] < i:
            checkpoints_fitness.append(last_fitness)
            checkpoint_idx += 1
        last_i = i
        last_fitness = fitness
    final_fitness = fitness_history[-1][1]
    if function_evaluations == checkpoints[-1]:
        while len(checkpoints_fitness) < len(checkpoints):
            checkpoints_fitness.append(final_fitness)
    return {
        "area_under_optimization_curve": area_under_optimization_curve
        / function_evaluations,
        "final_fitness": final_fitness,
        "checkpoints_fitness": checkpoints_fitness,
    }


def get_checkpoints(n_checkpoints: int, max_function_evaluations: int) -> np.ndarray:
    checkpoint_ratios = np.cumprod(
        np.full(shape=(n_checkpoints,), fill_value=CHECKPOINT_DIVISION_EXPONENT)
    )
    checkpoint_ratios = np.cumsum(checkpoint_ratios / checkpoint_ratios.sum())
    checkpoints = (checkpoint_ratios * max_function_evaluations).astype(int)
    checkpoints[-1] = (
        max_function_evaluations  # eliminate possibility of "error by one"
    )
    return checkpoints
