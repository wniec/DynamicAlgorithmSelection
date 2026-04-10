import numpy as np

MAX_DIM = 40


def get_runtime_stats(
    fitness_history: list[tuple[int, float]],
    function_evaluations: int,
    global_minimum: float,
) -> dict[str, float | list[float]]:
    """
    :param fitness_history: list of tuples [fe, fitness] with only points where best so far fitness improved
    :param function_evaluations: max number of function evaluations during run.
    :param global_minimum: the true optimal value of the problem to calculate relative precision.
    :return: dictionary of selected run statistics, ready to dump
    """
    area_under_optimization_curve = 0.0
    area_over_convergence_curve = 0.0
    last_i = 0

    # Define bounds for AOCClarge as used in the original code
    lb = 1e-8
    ub = 1e8

    # Precompute log bounds for logarithmic scaling
    log_lb = -8
    log_ub = 8

    for i, fitness in fitness_history:
        # Shift fitness by the global minimum so the area is always positive
        shifted_fitness = fitness - global_minimum

        # Linear Area Under Curve
        area_under_optimization_curve += shifted_fitness * (i - last_i)

        # Logarithmic Area Over Convergence Curve
        clipped_fitness = np.clip(shifted_fitness, lb, ub)
        normalized_fitness = (np.log10(clipped_fitness) - log_lb) / (log_ub - log_lb)

        # Correctly multiply the whole term by the width of the evaluation interval
        area_over_convergence_curve += (1.0 - normalized_fitness) * (i - last_i)

        last_i = i

    # Calculate the area for the final plateau
    final_shifted_fitness = fitness_history[-1][1] - global_minimum
    final_width = function_evaluations - fitness_history[-1][0]

    area_under_optimization_curve += final_shifted_fitness * final_width

    final_clipped = np.clip(final_shifted_fitness, lb, ub)
    final_normalized = (np.log10(final_clipped) - log_lb) / (log_ub - log_lb)
    area_over_convergence_curve += (1.0 - final_normalized) * final_width

    return {
        "area_under_optimization_curve": area_under_optimization_curve
        / function_evaluations,
        "aocc": area_over_convergence_curve / function_evaluations,
        "final_fitness": final_shifted_fitness,
    }


def get_extreme_stats(
    fitness_histories: dict[str, list[tuple[int, float]]],
    function_evaluations: int,
    global_minimum: float,
) -> tuple[dict[str, float | list[float]], dict[str, float | list[float]]]:
    """
    :param fitness_histories: list of lists of tuples [fe, fitness] with only points where best so far fitness improved for each algorithm
    :param function_evaluations: max number of function evaluations during run.
    :param global_minimum: the true optimal value of the problem.
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

    current_fitnesses = {
        alg: float("inf") for alg in fitness_histories
    }  # current best fitness for each algorithm
    current_worst_fitness = float("inf")  # worst performance so far for each algorithm

    worst_history = []
    for fe, algorithm, fitness in all_improvements:
        if fitness < current_fitnesses[algorithm]:
            current_fitnesses[algorithm] = fitness
            new_worst_fitness = max(
                i for i in current_fitnesses.values() if i != float("inf")
            )
            if new_worst_fitness < current_worst_fitness:
                worst_history.append((fe, fitness))
                current_worst_fitness = new_worst_fitness

    return (
        get_runtime_stats(best_history, function_evaluations, global_minimum),
        get_runtime_stats(worst_history, function_evaluations, global_minimum),
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
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class StepwiseRewardNormalizer:
    def __init__(self, max_steps, clip_reward=10.0):
        self.max_steps = max_steps
        self.clip = clip_reward
        self.stats = [RunningMeanStd(shape=()) for _ in range(max_steps + 1)]

    def normalize(self, reward, step_idx, update=True):
        idx = min(step_idx, self.max_steps - 1)

        mean = self.stats[idx].mean
        std = np.sqrt(self.stats[idx].var) + 1e-8
        normalized_reward = (reward - mean) / std

        if update:
            self.stats[idx].update(np.array([reward]))

        return np.clip(normalized_reward, -self.clip, self.clip)
