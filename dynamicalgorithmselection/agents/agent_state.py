import warnings

import numpy as np
import pandas as pd
from pflacco.classical_ela_features import (
    calculate_ela_meta,  # Meta-Model (Linear/Quadratic fit)
    calculate_nbc,  # Nearest Better Clustering
    calculate_dispersion,  # Dispersion of good solutions
    calculate_information_content,
    calculate_ela_distribution,  # Information Content
)
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from dynamicalgorithmselection.agents.agent_utils import MAX_DIM, RunningMeanStd

BASE_STATE_SIZE = 27
MAX_CONSIDERED_POPSIZE = 2500

ELA_FEATURES = [
    "ela_meta.lin_simple.coef.min",
    "ela_meta.lin_simple.coef.max",
    "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_simple.cond",
    "ela_meta.quad_w_interact.adj_r2",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
    "disp.ratio_mean_02",
    "disp.ratio_median_25",
    "disp.diff_mean_25",
    "disp.diff_median_02",
    "ic.h_max",
    "ic.eps_s",
    "ic.eps_max",
    "ic.m0",
    "ela_distr.skewness",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
]


def ela_state_representation(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        _, unique_indices = np.unique(x, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        x_deduplicated = x[unique_indices][-MAX_CONSIDERED_POPSIZE:]
        y_deduplicated = y[unique_indices][-MAX_CONSIDERED_POPSIZE:]

        x_raw = np.ascontiguousarray(x_deduplicated - x_deduplicated.mean()) / (
            x_deduplicated.std() + 1e-8
        )
        y_raw = np.ascontiguousarray(y_deduplicated - y_deduplicated.mean()) / (
            y_deduplicated.std() + 1e-8
        )

        x_norm = pd.DataFrame(x_raw).reset_index(drop=True)
        x_norm.columns = [f"x_{i}" for i in range(x_norm.shape[1])]
        y_norm = pd.Series(y_raw).reset_index(drop=True)

        is_unique = ~x_norm.duplicated()

        # If we lost data, re-slice to ensure alignment
        if not is_unique.all():
            x_norm = x_norm[is_unique].reset_index(drop=True)
            y_norm = y_norm[is_unique].reset_index(drop=True)

        if len(x_norm) < 50 or np.var(y_norm) < 1e-8:
            #  safeguard for degenerate case when population collapsed
            return np.zeros(22, dtype=np.float32)

        meta_feats = calculate_ela_meta(x_norm, y_norm)
        ela_distr = (
            calculate_ela_distribution(x_norm, y_norm)
            if ((y**2).sum() > 0 and np.var(y_norm) > 1e-8)
            else {
                i: 0.0
                for i in (
                    "ela_distr.skewness",
                    "ela_distr.kurtosis",
                    "ela_distr.number_of_peaks",
                )
            }
        )
        nbc_feats = calculate_nbc(x_norm, y_norm)
        disp_feats = calculate_dispersion(x_norm, y_norm)

        ic_feats = calculate_information_content(x_norm, y_norm)

        all_features = {
            **meta_feats,
            **nbc_feats,
            **disp_feats,
            **ic_feats,
            **ela_distr,
        }
        return np.array([all_features[i] for i in ELA_FEATURES], dtype=np.float32)


class AgentState:
    def __init__(
        self,
        n_actions,
        choice_history,
        n_checkpoints,
        n_dim_problem,
    ):
        self.n_actions = n_actions
        self.n_checkpoints = n_checkpoints
        self.ndim_problem = n_dim_problem
        self.choice_history = choice_history
        if len(choice_history) < 1:
            return  # the rest of properties won't be needed

        self.last_action_index = (
            self.choice_history[-1] if self.choice_history else None
        )
        self.last_action_encoded = [0 for _ in range(n_actions)]

        if self.last_action_index is not None:
            self.last_action_encoded[self.last_action_index] = 1

        self.choices_frequency = [
            sum(1 for i in self.choice_history if i == j)
            / (len(self.choice_history) or 1)
            for j in range(self.n_actions)
        ]

    def choice_entropy(self) -> float:
        return -(
            np.array(self.choices_frequency)
            * np.nan_to_num(np.log(self.choices_frequency), neginf=0, posinf=0, nan=0)
        ).sum() / np.log(len(self.choices_frequency))

    def same_action_counter(self) -> int:
        same_action_counter = 0
        for i in reversed(self.choice_history):
            if i == self.last_action_index:
                same_action_counter += 1
            else:
                break
        return same_action_counter

    def get_initial_state(self):
        vector = [
            *(0.0 for _ in range(self.n_actions)),  # last action encoded
            0.0,  # same action counter
            *(0.0 for _ in range(self.n_actions)),  # choices frequency
            0.0,  # choice entropy
            self.ndim_problem / MAX_DIM,  # normalized problem dimension
        ]
        return np.array(vector, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        if len(self.choice_history) < 1:
            return self.get_initial_state()
        else:
            vector = [
                *self.last_action_encoded,
                self.same_action_counter() / self.n_checkpoints,
                *self.choices_frequency,
                self.choice_entropy(),
                self.ndim_problem / MAX_DIM,
            ]
        return np.array(vector, dtype=np.float32)


class StateNormalizer:
    def __init__(self, input_shape):
        self.rms = RunningMeanStd(shape=input_shape)

    def normalize(self, state, update=True):
        """
        Normalizes the state: (state - mean) / std.

        Args:
            state (np.array): The input state vector.
            update (bool): Whether to update the running statistics.
                           Usually True during training, False during testing.
        """
        state = np.asarray(state)

        if update:
            if len(state.shape) == 1:
                self.rms.update(state.reshape(1, -1))
            else:
                self.rms.update(state)

        std = np.sqrt(self.rms.var) + 1e-8

        normalized_state = (state - self.rms.mean) / std
        return np.clip(normalized_state, -5.0, 5.0)


def negative_slope_coefficient(group_cost, sample_cost):  # [j]
    gs = sample_cost.shape[0]
    m = 10
    gs -= gs % m  # to be divisible
    if gs < m:  # not enough costs for m dividing
        return 0
    sorted_cost = np.array(sorted(list(zip(group_cost[:gs], sample_cost[:gs]))))
    sorted_group = sorted_cost[:, 0].reshape(m, -1)
    sorted_sample = sorted_cost[:, 1].reshape(m, -1)
    Ms = np.mean(sorted_group, -1)
    Ns = np.mean(sorted_sample, -1)
    nsc = np.minimum((Ns[1:] - Ns[:-1]) / (Ms[1:] - Ms[:-1] + 1e-8), 0)
    return np.sum(nsc)


def get_la_features(agent, pop_x, pop_y):
    """
    Extracts 9 Landscape Analysis features based on the logic in Population.py.
    Uses a single-step random walk for sampling-based features (f5-f8) to
    save function evaluations.
    """
    n = len(pop_x)

    best_y = np.min(pop_y)
    best_x = pop_x[np.argmin(pop_y)]
    norm_factor = (
        agent.initial_cost
        if hasattr(agent, "initial_cost")
        and agent.initial_cost
        and abs(agent.initial_cost) > 1e-9
        else 1.0
    )
    f1_gbc = best_y / norm_factor

    dists_to_best = np.linalg.norm(pop_x - best_x, axis=1)
    if np.std(pop_y) < 1e-9 or np.std(dists_to_best) < 1e-9:
        f2_fdc = 0.0
    else:
        fdc, _ = spearmanr(pop_y, dists_to_best)
        f2_fdc = fdc if not np.isnan(fdc) else 0.0

    n_top = max(2, int(0.1 * n))
    if n > 1:
        dist_matrix_all = pdist(pop_x)
        disp_all = np.mean(dist_matrix_all) if len(dist_matrix_all) > 0 else 0.0

        # Get distances for the top 10% individuals
        top_idx = np.argsort(pop_y)[:n_top]
        dist_matrix_top = pdist(pop_x[top_idx])
        disp_top = np.mean(dist_matrix_top) if len(dist_matrix_top) > 0 else 0.0

        f3_disp = disp_all - disp_top
        f4_disp_ratio = disp_top / disp_all if disp_all > 1e-9 else 0.0
    else:
        f3_disp, f4_disp_ratio = 0.0, 0.0

    # Adjust step size based on your search space bounds if available
    step_scale = 0.01
    if hasattr(agent, "Xmax") and hasattr(agent, "Xmin"):
        step_size = step_scale * (agent.Xmax - agent.Xmin)
    else:
        step_size = step_scale

    random_walk_samples = pop_x + np.random.normal(0, step_size, size=pop_x.shape)

    # Evaluate the random walk samples
    sample_costs = np.array([agent.fitness_function(i) for i in random_walk_samples])
    agent.n_function_evaluations += n  # Increment evaluations by population size

    # Calculate differences between the walk and the current population
    diffs = np.array(sample_costs) - pop_y

    # --- Feature 5: Negative Slope Coefficient (nsc) ---
    # Proportion of steps that resulted in an improvement
    f5_nsc = negative_slope_coefficient(pop_y, sample_cost=sample_costs)

    # --- Feature 6: Average Neutral Ratio (anr) ---
    # Proportion of steps that resulted in practically zero change
    eps = 1e-8
    f6_anr = np.sum(np.abs(diffs) < eps) / n

    f7_ni = np.sum(diffs >= 0) / n  # Ratio of individuals that failed to improve
    f8_nw = np.sum(diffs <= 0) / n  # Ratio of individuals that failed to worsen

    # --- Feature 9: Progress ---
    f9_progress = agent.n_function_evaluations / agent.max_function_evaluations

    return np.array(
        [
            f1_gbc,
            f2_fdc,
            f3_disp,
            f4_disp_ratio,
            f5_nsc,
            f6_anr,
            f7_ni,
            f8_nw,
            f9_progress,
        ]
    )
