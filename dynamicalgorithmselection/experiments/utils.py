import json
import os
from itertools import islice, product
from typing import Type

import cocoex
import numpy as np

from dynamicalgorithmselection.agents.agent_utils import (
    get_checkpoints,
    get_runtime_stats,
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


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


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
