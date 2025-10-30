from itertools import product
from typing import Type

import cocoex
import numpy as np
import torch
from tqdm import tqdm
from dynamicalgorithmselection.agent import Agent
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


def coco_bbob(
    optimizer: Type[Optimizer],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    train: bool = True,
    easy_mode: bool = True,
):
    suite, output = "bbob", name
    observer = cocoex.Observer(suite, "result_folder: " + output)
    cocoex.utilities.MiniPrint()
    function_ids = (
        EASY_TRAIN_BBOB
        if ((easy_mode and train) or (not easy_mode and not train))
        else ALL_FUNCTIONS.difference(EASY_TRAIN_BBOB)
    )
    problems_suite = cocoex.Suite(suite, "", "")
    problem_ids = [
        f"bbob_f{f_id:03d}_i{i_id:02d}_d{dim:02d}"
        for i_id, f_id, dim in product(INSTANCE_IDS, function_ids, DIMENSIONS)
    ]
    agent_state = {}
    for problem_id in tqdm(np.random.permutation(problem_ids)):
        problem_instance = problems_suite.get_problem(problem_id)
        problem_instance.observe_with(observer)
        options["max_function_evaluations"] = (
            evaluations_multiplier * problem_instance.dimension
        )
        options.update(agent_state)
        options["train"] = train
        options["verbose"] = False
        if train:
            results, agent_state = coco_bbob_single_function(
                optimizer, problem_instance, options
            )
        else:
            coco_bbob_single_function(optimizer, problem_instance, options)
    if train:
        torch.save(agent_state, f"{name}.pth")
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
    # run black-box optimizer
    results = optimizer(problem, options).optimize()
    return results
