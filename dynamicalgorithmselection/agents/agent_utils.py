import numpy as np
import torch
from torch import nn
import torch.nn.init as init

GAMMA = 0.9
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
    def __init__(self, n_actions: int, dropout_p: float = 0.35):
        super().__init__()
        # Replace LSTM with a standard MLP feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(BASE_STATE_SIZE + n_actions * 2, HIDDEN_SIZE),
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
    def __init__(self, n_actions: int, dropout_p: float = 0.35):
        super().__init__()
        # Replace LSTM with a standard MLP feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(BASE_STATE_SIZE + n_actions * 2, HIDDEN_SIZE),
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
    checkpoints: np.ndarray,
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
    area_under_optimization_curve += fitness_history[-1][1] * (
        function_evaluations - fitness_history[-1][0]
    )
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


def get_extreme_stats(
    fitness_histories: dict[str, list[tuple[int, float]]],
    function_evaluations: int,
    checkpoints: np.ndarray,
) -> tuple[dict[str, float | list[float]], dict[str, float | list[float]]]:
    """
    :param fitness_histories: list of lists of tuples [fe, fitness] with only points where best so far fitness improved for each algorithm
    :param function_evaluations: max number of function evaluations during run.
    :param checkpoints: list of checkpoints by their n_function_evaluations
    :return: dictionary of selected run statistics, ready to dump
    """
    all_improvements = []
    for algorithm, run in fitness_histories.items():
        for fe, fitness in run:
            all_improvements.append((fe, algorithm, fitness))

    all_improvements.sort(
        key=lambda x: (x[0], -x[2])
    )  # sort by fe - increasing and fitness - increasing
    current_fitness = float("inf")

    best_history = []
    for fe, _, fitness in all_improvements:
        if fitness < current_fitness:
            current_fitness = fitness
            best_history.append((fe, fitness))

    all_improvements.sort(
        key=lambda x: (x[0], -x[2])
    )  # sort fe - increasing and by fitness - decreasing

    current_fitness = {
        alg: float("inf") for alg in fitness_histories
    }  # current best fitness for each algorithm
    current_worst_fitness = float("inf")  # worst performance so far for each algorithm

    worst_history = []
    for fe, algorithm, fitness in all_improvements:
        if fitness < current_fitness[algorithm]:
            current_fitness[algorithm] = fitness
            new_worst_fitness = max(
                i for i in current_fitness.values() if i != float("inf")
            )
            if new_worst_fitness < current_worst_fitness:
                worst_history.append((fe, fitness))
                current_worst_fitness = new_worst_fitness

    return (
        get_runtime_stats(best_history, function_evaluations, checkpoints),
        get_runtime_stats(worst_history, function_evaluations, checkpoints),
    )


def get_checkpoints(
    n_checkpoints: int, max_function_evaluations: int, n_individuals: int, cdb: float
) -> np.ndarray:
    checkpoint_ratios = np.cumprod(np.full(shape=(n_checkpoints,), fill_value=cdb))
    checkpoint_ratios = np.cumsum(checkpoint_ratios / checkpoint_ratios.sum())
    checkpoints = (checkpoint_ratios * max_function_evaluations).astype(int)
    checkpoints[-1] = (
        max_function_evaluations  # eliminate possibility of "error by one"
    )
    checkpoints[0] = max(checkpoints[0], n_individuals)
    for i in range(1, len(checkpoints)):
        checkpoints[i] = max(checkpoints[i - 1] + n_individuals, checkpoints[i])
    return checkpoints


class RunningMeanStd:
    def __init__(self, n_actions: int, epsilon=1e-4):
        shape = (BASE_STATE_SIZE + n_actions * 2,)
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        # Expects x to be shape (batch_size, n_features)
        self.update(x)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


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
