import json
import os
import pickle
import re
from itertools import product, batched, cycle
from typing import Type, Optional

import cocoex
import numpy as np
import neat
from tqdm import tqdm

from dynamicalgorithmselection.agents.agent_utils import (
    BASE_STATE_SIZE,
    get_runtime_stats,
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


def dump_stats(
    results,
    name,
    problem_instance,
    max_function_evaluations,
    n_checkpoints,
):
    checkpoints = get_checkpoints(n_checkpoints, max_function_evaluations)
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
    elif options["baselines"]:
        # running only baselines
        return _coco_bbob_test_all(optimizer, options, evaluations_multiplier, mode)
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


def eval_genomes(
    genomes,
    config,
    optimizer,
    problem_batches,
    suite,
    options,
    evaluations_multiplier,
):
    run = options.get("run", None)
    num_actions = len(options.get("action_space"))
    for problem_batch, (genome_id, genome) in zip(
        cycle(problem_batches), tqdm(genomes)
    ):
        fitness = 0
        actions = []
        for problem_id in problem_batch:
            problem_instance = suite.get_problem(problem_id)
            options["max_function_evaluations"] = (
                evaluations_multiplier * problem_instance.dimension
            )
            options["train_mode"] = True
            options["verbose"] = False
            options["net"] = neat.nn.FeedForwardNetwork.create(genome, config)
            results = coco_bbob_single_function(optimizer, problem_instance, options)
            fitness += results["mean_reward"]
            actions.extend(results["actions"])

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

        genome.fitness = fitness / len(problem_batch) + norm_entropy / 35
        if run is not None:
            run.log({"fitness": genome.fitness})
            run.log({"entropy": norm_entropy})


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
    problems_suite, cv_folds = get_cv_folds(5)
    observer = cocoex.Observer("bbob", "result_folder: " + options.get("name"))
    name = options["name"]
    for i, (train_set, test_set) in enumerate(cv_folds):
        options["name"] = f"{name}_cv{i}"
        print(f"Running cross validation training, fold {i + 1}")
        run_training(optimizer, options, evaluations_multiplier, problems_suite, problem_ids=train_set)
        print(f"Running cross validation testing, fold {i + 1}")
        run_testing(optimizer, options, evaluations_multiplier, problems_suite, problem_ids=test_set, observer=observer)
    return observer.result_folder

def get_cv_folds(n: int):
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
        selected = np.random.choice(list(remaining_problem_ids), size=len(all_problem_ids) // n, replace=False).tolist()
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
    run_training(optimizer, options, evaluations_multiplier, problems_suite, problem_ids)


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
    problems_suite, problem_ids = get_suite(mode, True)
    batch_size = 30
    adjust_config(
        2 * len(options.get("action_space")) + BASE_STATE_SIZE,
        len(options.get("action_space")),
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neuroevolution_config",
    )
    # Create the population, which is the top-level object for a NEAT run.
    population = neat.Population(config)
    problems_batches = list(batched((np.random.permutation(problem_ids)), batch_size))
    winner = population.run(
        lambda genomes, config: eval_genomes(
            genomes,
            config,
            optimizer,
            problems_batches,
            problems_suite,
            options,
            evaluations_multiplier,
        ),
        300,
    )
    with open(
        os.path.join("models", f"DAS_train_{options.get('name')}.pkl"), "wb"
    ) as f:
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
    run_testing(optimizer, options, evaluations_multiplier, problems_suite, problem_ids, observer)
    return observer.result_folder


def _coco_bbob_test_all(optimizer, options, evaluations_multiplier, mode):
    results_dir = os.path.join("results", f"{options.get('name')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite("baselines", False)
    observer = cocoex.Observer("bbob", "result_folder: " + options.get("name"))
    run_testing(optimizer, options, evaluations_multiplier, problems_suite, problem_ids, observer)
    return observer.result_folder


def run_testing(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int,
    problems_suite: cocoex.Suite,
    problem_ids: list[str],
    observer: cocoex.Observer
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
