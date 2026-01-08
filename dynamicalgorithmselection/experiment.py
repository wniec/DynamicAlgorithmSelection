import json
import os
import pickle
import re
import random
import multiprocessing
from itertools import product, cycle
from typing import Type, Optional, Any
from .utils import batched

import cocoex
import numpy as np
import neat
from tqdm import tqdm
import wandb

from dynamicalgorithmselection.agents.agent_utils import (
    get_runtime_stats,
    get_extreme_stats,
    get_checkpoints,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer

"""
**BBOB (F1-F24)**
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy**        | 4, 6-14, 18-20, 22-24 | 1, 2, 3, 5, 15, 16, 17, 21 |
| **difficult**   | 1, 2, 3, 5, 15, 16, 17, 21 | 4, 6-14, 18-20, 22-24 |
"""

EASY_TRAIN_BBOB = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}
ALL_FUNCTIONS = {_ for _ in range(1, 25)}
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
DIMENSIONS = [2, 3, 5, 10, 20, 40]


def _eval_wrapper(eval_function, seed, genome, config):
    """Wrapper that sets deterministic per-genome seed before evaluation."""
    if seed is not None:
        random.seed(seed + genome.key)
    return eval_function(genome, config)


class ParallelEvaluator:
    def __init__(
        self,
        num_workers,
        eval_function,
        timeout=None,
        initializer=None,
        initargs=(),
        maxtasksperchild=None,
        seed=None,
    ):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.seed = seed
        self.initializer = initializer
        self.initargs = initargs
        self.maxtasksperchild = maxtasksperchild
        self.pool = multiprocessing.Pool(
            processes=num_workers,
            maxtasksperchild=maxtasksperchild,
            initializer=initializer,
            initargs=initargs,
        )
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self.pool is not None and not self._closed:
            self._closed = True
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close()

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            if self.seed is not None:
                jobs.append(
                    self.pool.apply_async(
                        _eval_wrapper, (self.eval_function, self.seed, genome, config)
                    )
                )
            else:
                jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        for job, (ignored_genome_id, genome) in tqdm(
            zip(jobs, genomes), total=len(jobs), desc="Evaluating Genomes"
        ):
            # Result is now a tuple: (fitness, log_dict)
            fitness, log_data = job.get(timeout=self.timeout)
            genome.fitness = fitness

            # Log to wandb in the main process
            if log_data and wandb.run is not None:
                wandb.log(log_data)


class GenomeEvaluator:
    """
    Functor class to evaluate a single genome.
    Allows pickling of configuration/options for multiprocessing.
    """

    def __init__(self, optimizer, options, evaluations_multiplier):
        self.optimizer = optimizer
        self.evaluations_multiplier = evaluations_multiplier
        # Create a clean copy of options (remove unpicklable loggers)
        self.options = options.copy()
        if "run" in self.options:
            del self.options["run"]

    def __call__(self, genome, config):
        # Re-initialize suite inside worker to avoid pickling pointers
        cocoex.utilities.MiniPrint()
        suite = cocoex.Suite("bbob", "", "")

        if not hasattr(genome, "problem_batch"):
            return 0.0, {}

        fitness = 0
        actions = []
        num_actions = len(self.options.get("action_space"))
        local_options = self.options.copy()

        for problem_id in genome.problem_batch:
            problem_instance = suite.get_problem(problem_id)
            local_options["max_function_evaluations"] = (
                self.evaluations_multiplier * problem_instance.dimension
            )
            local_options["train_mode"] = True
            local_options["verbose"] = False
            local_options["net"] = neat.nn.FeedForwardNetwork.create(genome, config)

            results = coco_bbob_single_function(
                self.optimizer, problem_instance, local_options
            )

            fitness += results["mean_reward"]
            local_options["state_normalizer"] = results["state_normalizer"]
            local_options["reward_normalizer"] = results["reward_normalizer"]
            actions.extend(results["actions"])

            problem_instance.free()

        # Calculate Entropy
        choices_count = np.array(
            [
                sum(1 for i in actions if i == j) / (len(actions) or 1)
                for j in range(len(actions))
            ]
        )
        norm_entropy = -(
            choices_count
            * np.nan_to_num(np.log(choices_count), nan=0, neginf=0, posinf=0)
        ).sum() / np.log(num_actions)

        final_fitness = fitness / len(genome.problem_batch) + norm_entropy / 35

        # Return fitness AND data to log
        log_data = {
            "fitness": final_fitness,
            "entropy": norm_entropy,
            "mean_reward": fitness / len(genome.problem_batch),
        }
        return final_fitness, log_data


def dump_stats(
    results,
    name,
    problem_instance,
    max_function_evaluations,
    n_checkpoints,
    n_individuals,
    cdb,
):
    checkpoints = get_checkpoints(
        n_checkpoints, max_function_evaluations, n_individuals, cdb
    )
    with open(
        os.path.join(
            "results",
            f"{name}",
            f"{problem_instance}.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                problem_instance: get_runtime_stats(
                    results["fitness_history"],
                    max_function_evaluations,
                    checkpoints,
                )
            },
            f,
        )


def dump_extreme_stats(
    optimizer_portfolio: list[Type[Optimizer]],
    stats,
    problem_instance,
    max_function_evaluations,
    n_checkpoints,
    n_individuals,
    cdb,
):
    checkpoints = get_checkpoints(
        n_checkpoints, max_function_evaluations, n_individuals, cdb
    )
    best_case, worst_case = get_extreme_stats(
        stats, max_function_evaluations, checkpoints
    )
    portfolio_name = "_".join(i.__name__ for i in optimizer_portfolio)
    with open(
        os.path.join(
            "results",
            f"{portfolio_name}_best",
            f"{problem_instance}.json",
        ),
        "w",
    ) as f:
        json.dump(
            {problem_instance: best_case},
            f,
        )
    with open(
        os.path.join(
            "results",
            f"{portfolio_name}_worst",
            f"{problem_instance}.json",
        ),
        "w",
    ) as f:
        json.dump(
            {problem_instance: worst_case},
            f,
        )


def coco_bbob_experiment(
    optimizer: Type[Optimizer],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    train: bool = True,
    mode: str = "easy",
    agent: Optional[str] = "policy-gradient",
):
    options["name"] = name
    if mode == "CV":
        return run_cross_validation(optimizer, options, evaluations_multiplier)
    elif agent == "random":
        # running random baseline
        return _coco_bbob_test_all(optimizer, options, evaluations_multiplier, mode)
    elif options.get("baselines"):
        # running only baselines
        return run_comparison(
            options.get("optimizer_portfolio"), options, evaluations_multiplier
        )
    elif not train:
        return _coco_bbob_test(optimizer, options, evaluations_multiplier, mode)
    elif agent == "neuroevolution":
        return _coco_bbob_neuroevolution_train(
            optimizer, options, evaluations_multiplier, mode
        )
    else:
        return _coco_bbob_policy_gradient_train(
            optimizer, options, evaluations_multiplier, mode
        )


def get_suite(mode, train):
    """
    :param mode:  mode of the training (LOPO: easy and hard) or LOIO
    :param train: if suite should be for testing or training:
    :return suite and list of problem ids:
    """
    cocoex.utilities.MiniPrint()
    problems_suite = cocoex.Suite("bbob", "", "")
    all_problem_ids = [
        f"bbob_f{f_id:03d}_i{i_id:02d}_d{dim:02d}"
        for i_id, f_id, dim in product(INSTANCE_IDS, ALL_FUNCTIONS, DIMENSIONS)
    ]
    if mode in ["easy", "hard"]:
        easy = mode == "easy"
        function_ids = (
            EASY_TRAIN_BBOB
            if ((easy and train) or (not easy and not train))
            else ALL_FUNCTIONS.difference(EASY_TRAIN_BBOB)
        )

        problem_ids = [
            f"bbob_f{f_id:03d}_i{i_id:02d}_d{dim:02d}"
            for i_id, f_id, dim in product(INSTANCE_IDS, function_ids, DIMENSIONS)
        ]

    elif mode == "LOIO":
        with open("LOIO_train_set.json") as f:
            problem_ids = json.load(f)["data"]
        np.random.seed(1234)
        if train:
            pass
        else:
            problem_ids = list(set(all_problem_ids).difference(problem_ids))
    elif mode == "CV":
        raise ValueError("CV mode is not suitable for get_suite function")
    else:
        return problems_suite, all_problem_ids
    return problems_suite, problem_ids


def run_cross_validation(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
):
    results_dir = os.path.join("results", f"{options.get('name')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    cocoex.utilities.MiniPrint()
    problems_suite, cv_folds = _get_cv_folds(4)
    observer = cocoex.Observer("bbob", "result_folder: " + options.get("name"))
    for i, (train_set, test_set) in enumerate(cv_folds):
        print(f"Running cross validation training, fold {i + 1}")
        run_training(
            optimizer,
            options,
            evaluations_multiplier,
            problems_suite,
            problem_ids=train_set,
        )
        print(f"Running cross validation testing, fold {i + 1}")
        run_testing(
            optimizer,
            options,
            evaluations_multiplier,
            problems_suite,
            problem_ids=test_set,
            observer=observer,
        )
        options.update(
            {
                "buffer": None,
                "model_parameters": None,
                "optimizer": None,
                "reward_normalizer": None,
                "state_normalizer": None,
            }
        )
    return observer.result_folder


def _get_cv_folds(n: int):
    """
    :param n:  number of cross validation folds
    :return suite, list of (train set, test set) pairs:
    """
    np.random.seed(1234)
    cocoex.utilities.MiniPrint()
    problems_suite = cocoex.Suite("bbob", "", "")
    all_problem_ids = [
        f"bbob_f{f_id:03d}_i{i_id:02d}_d{dim:02d}"
        for i_id, f_id, dim in product(INSTANCE_IDS, ALL_FUNCTIONS, DIMENSIONS)
    ]
    remaining_problem_ids = set(all_problem_ids)
    test_sets = []
    for i in range(n):
        selected = np.random.choice(
            list(remaining_problem_ids), size=len(all_problem_ids) // n, replace=False
        ).tolist()
        test_sets.append(selected)
        remaining_problem_ids = remaining_problem_ids.difference(selected)
    return problems_suite, [
        (list(set(all_problem_ids).difference(test_set)), test_set)
        for test_set in test_sets
    ]


def run_training(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int,
    problems_suite: cocoex.Suite,
    problem_ids: list[str],
):
    agent_state = {}
    for problem_id in tqdm(np.random.permutation(problem_ids)):
        problem_instance = problems_suite.get_problem(problem_id)
        max_fe = evaluations_multiplier * problem_instance.dimension
        options["max_function_evaluations"] = max_fe
        options.update(agent_state)
        options["train_mode"] = True
        options["verbose"] = False
        results, agent_state = coco_bbob_single_function(
            optimizer, problem_instance, options
        )
        options["buffer"] = agent_state["buffer"]
        problem_instance.free()


def _coco_bbob_policy_gradient_train(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    mode: str = "easy",
):
    results_dir = os.path.join("results", f"{options.get('name')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite(mode, True)
    run_training(
        optimizer, options, evaluations_multiplier, problems_suite, problem_ids
    )


def adjust_config(n_inputs, n_outputs):
    with open("neuroevolution_config", "r") as f:
        config_content = f.read()

    config_content = re.sub(
        pattern="num_inputs.*", repl=f"num_inputs = {n_inputs}", string=config_content
    )
    config_content = re.sub(
        pattern="num_outputs.*",
        repl=f"num_outputs = {n_outputs}",
        string=config_content,
    )

    with open("neuroevolution_config", "w") as f:
        f.write(config_content)


def _coco_bbob_neuroevolution_train(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    mode: str = "easy",
):
    cocoex.utilities.MiniPrint()
    _, problem_ids = get_suite(mode, True)
    batch_size = 30

    adjust_config(
        16 * options.get("n_individuals"),
        len(options.get("action_space")),
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neuroevolution_config",
    )

    population = neat.Population(config)

    all_problem_batches = list(
        batched((np.random.permutation(problem_ids)), batch_size)
    )
    batch_cycler = cycle(all_problem_batches)

    # Pre-assign batches
    for genome_id, genome in population.population.items():
        genome.problem_batch = next(batch_cycler)

    num_workers = len(os.sched_getaffinity(0)) - 1
    print(f"Using {num_workers} workers.")

    evaluator_func = GenomeEvaluator(optimizer, options, evaluations_multiplier)

    with ParallelEvaluator(num_workers, evaluator_func) as pe:
        winner = population.run(pe.evaluate, 20)

    with open(os.path.join("models", f"{options.get('name')}.pkl"), "wb") as f:
        pickle.dump(winner, f)


def _coco_bbob_test(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    mode: str = "easy",
):
    results_dir = os.path.join("results", f"{options.get('name')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite(mode, False)
    observer = cocoex.Observer("bbob", "result_folder: " + options.get("name"))
    run_testing(
        optimizer,
        options,
        evaluations_multiplier,
        problems_suite,
        problem_ids,
        observer,
    )
    return observer.result_folder


def _coco_bbob_test_all(optimizer, options, evaluations_multiplier, mode):
    results_dir = os.path.join("results", f"{options.get('name')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite("baselines", False)
    observer = cocoex.Observer("bbob", "result_folder: " + options.get("name"))
    run_testing(
        optimizer,
        options,
        evaluations_multiplier,
        problems_suite,
        problem_ids,
        observer,
    )
    return observer.result_folder


def run_testing(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int,
    problems_suite: cocoex.Suite,
    problem_ids: list[str],
    observer: cocoex.Observer,
):
    for problem_id in tqdm(problem_ids):
        problem_instance = problems_suite.get_problem(problem_id)
        problem_instance.observe_with(observer)
        max_fe = evaluations_multiplier * problem_instance.dimension
        options["max_function_evaluations"] = max_fe
        options["train_mode"] = False
        options["verbose"] = False
        results = coco_bbob_single_function(optimizer, problem_instance, options)
        problem_instance.free()
        dump_stats(
            results[0] if isinstance(results, tuple) else results,
            options.get("name"),
            problem_id,
            max_fe,
            options.get("n_checkpoints"),
            options.get("n_individuals"),
            options.get("cdb"),
        )


def run_comparison(
    optimizer_portfolio: list[Type[Optimizer]],
    options: dict,
    evaluations_multiplier: int,
):
    for optimizer in optimizer_portfolio:
        results_dir = os.path.join("results", f"{optimizer.__name__}")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
    best_dir = os.path.join(
        "results", "_".join(i.__name__ for i in optimizer_portfolio) + "_best"
    )
    if not os.path.exists(best_dir):
        os.mkdir(best_dir)
    worst_dir = os.path.join(
        "results", "_".join(i.__name__ for i in optimizer_portfolio) + "_worst"
    )
    if not os.path.exists(worst_dir):
        os.mkdir(worst_dir)
    cocoex.utilities.MiniPrint()
    _, problem_ids = get_suite("all", False)
    suites = {
        optimizer.__name__: get_suite("all", False)[0]
        for optimizer in optimizer_portfolio
    }
    for problem_id in tqdm(problem_ids):
        max_fe = None
        stats = {}
        for optimizer in optimizer_portfolio:
            problem_instance = suites[optimizer.__name__].get_problem(problem_id)
            max_fe = evaluations_multiplier * problem_instance.dimension

            options["max_function_evaluations"] = max_fe
            options["train_mode"] = False
            options["verbose"] = False
            results = coco_bbob_single_function(optimizer, problem_instance, options)
            problem_instance.free()
            stats[optimizer.__name__] = results["fitness_history"]
            dump_stats(
                results[0] if isinstance(results, tuple) else results,
                optimizer.__name__,
                problem_id,
                max_fe,
                options.get("n_checkpoints"),
                options.get("n_individuals"),
                options.get("cdb"),
            )

        dump_extreme_stats(
            optimizer_portfolio,
            stats,
            problem_id,
            max_fe,
            options.get("n_checkpoints"),
            options.get("n_individuals"),
            options.get("cdb"),
        )


def coco_bbob_single_function(
    optimizer: Type[Optimizer], function: cocoex.Problem, options
):
    problem = {
        "fitness_function": function,
        "ndim_problem": function.dimension,
        "lower_boundary": function.lower_bounds,
        "upper_boundary": function.upper_bounds,
    }
    results = optimizer(problem, options).optimize()
    return results
