import pickle
from itertools import product, batched, cycle
from typing import Type

import cocoex
import numpy as np
import neat
import torch
from tqdm import tqdm
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


def coco_bbob_experiment(
    optimizer: Type[Optimizer],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    train: bool = True,
    easy_mode: bool = True,
    neuroevolution: bool = False,
):
    if not train:
        return _coco_bbob_test(
            optimizer, options, name, evaluations_multiplier, easy_mode
        )
    elif neuroevolution:
        return _coco_bbob_neuroevolution_train(
            optimizer, options, name, evaluations_multiplier, easy_mode
        )
    else:
        return _coco_bbob_policy_gradient_train(
            optimizer, options, name, evaluations_multiplier, easy_mode
        )


def eval_genomes(
    genomes,
    config,
    optimizer,
    problem_batches,
    suite,
    options,
    observer,
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
            problem_instance.observe_with(observer)
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


def get_suite(name, easy_mode, train):
    suite, output = "bbob", name
    cocoex.utilities.MiniPrint()
    problems_suite = cocoex.Suite(suite, "", "")
    observer = cocoex.Observer(suite, "result_folder: " + output)
    function_ids = (
        EASY_TRAIN_BBOB
        if ((easy_mode and train) or (not easy_mode and not train))
        else ALL_FUNCTIONS.difference(EASY_TRAIN_BBOB)
    )

    problem_ids = [
        f"bbob_f{f_id:03d}_i{i_id:02d}_d{dim:02d}"
        for i_id, f_id, dim in product(INSTANCE_IDS, function_ids, DIMENSIONS)
    ]
    return problems_suite, problem_ids, observer


def _coco_bbob_policy_gradient_train(
    optimizer: Type[Optimizer],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    easy_mode: bool = True,
):
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids, observer = get_suite(name, easy_mode, True)
    agent_state = {}
    for problem_id in tqdm(np.random.permutation(problem_ids)):
        problem_instance = problems_suite.get_problem(problem_id)
        problem_instance.observe_with(observer)
        options["max_function_evaluations"] = (
            evaluations_multiplier * problem_instance.dimension
        )
        options.update(agent_state)
        options["train_mode"] = True
        options["verbose"] = False
        results, agent_state = coco_bbob_single_function(
            optimizer, problem_instance, options
        )
        problem_instance.free()
    torch.save(agent_state, f"{name}.pth")
    return observer.result_folder


def _coco_bbob_neuroevolution_train(
    optimizer: Type[Optimizer],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    easy_mode: bool = True,
):
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids, observer = get_suite(name, easy_mode, True)
    batch_size = 30
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config",
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
            observer,
            evaluations_multiplier,
        ),
        300,
    )
    with open(f"DAS_train_{name}.pkl", "wb") as f:
        pickle.dump(winner, f)
    return observer.result_folder


def _coco_bbob_test(
    optimizer: Type[Optimizer],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    easy_mode: bool = True,
):
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids, observer = get_suite(name, easy_mode, False)
    for problem_id in tqdm(problem_ids):
        problem_instance = problems_suite.get_problem(problem_id)
        problem_instance.observe_with(observer)
        options["max_function_evaluations"] = (
            evaluations_multiplier * problem_instance.dimension
        )
        options["train_mode"] = False
        options["verbose"] = False
        coco_bbob_single_function(optimizer, problem_instance, options)
        problem_instance.free()
    return observer.result_folder


def coco_bbob_single_function(
    optimizer: Type[Optimizer], function: cocoex.interface.Problem, options
):
    problem = {
        "fitness_function": function,
        "ndim_problem": function.dimension,
        "lower_boundary": function.lower_bounds,
        "upper_boundary": function.upper_bounds,
    }
    results = optimizer(problem, options).optimize()
    return results
