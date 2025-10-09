import json
from itertools import product

import cocoex
import cocopp
import numpy as np
import torch
from tqdm import tqdm

from agent import Agent
from optimizers.G3PCX import G3PCX
from optimizers.Optimizer import Optimizer
from optimizers.SPSO import SPSO
from optimizers.LMCMAES import LMCMAES

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
    optimizer: Optimizer,
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    train: bool = True,
    easy_mode: bool = True,
):
    all_actor_losses = []
    all_critic_losses = []
    if train:
        agent_state = {i: None for i in ('actor_params', 'critic_params', 'actor_optimizer', 'critic_optimizer')}

    else:
        agent_state = torch.load("state.pth")
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
            all_actor_losses.extend(results["actor_losses"])
            all_critic_losses.extend(results["critic_losses"])
        else:
            coco_bbob_single_function(optimizer, problem_instance, options)
    if train:
        torch.save(agent_state, "state.pth")
        with open("actor_losses.json", "w") as file:
            json.dump(all_actor_losses, file)
        with open("critic_losses.json", "w") as file:
            json.dump(all_critic_losses, file)
    return observer.result_folder


def coco_bbob_single_function(
    optimizer: Optimizer, function: cocoex.interface.Problem, options
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


if __name__ == "__main__":
    n = 20
    multiplier = 10_000
    coco_bbob(
        Agent,
        {"sub_optimization_ratio": 10, "n_individuals": n},
        name="DAS_train",
        evaluations_multiplier=multiplier,
        train=True,
    )
    coco_bbob(
        Agent,
        {"sub_optimization_ratio": 10, "n_individuals": n},
        name="DAS_test",
        evaluations_multiplier=multiplier,
        train=False,
    )
    """coco_bbob(
        LMCMAES,
        {"n_individuals": n},
        name="LMCMAES",
        evaluations_multiplier=multiplier,
        train=False,
    )"""
    """coco_bbob(
        G3PCX,
        {"n_individuals": n},
        name="G3PCX",
        evaluations_multiplier=multiplier,
        train=False,
    )"""
    """coco_bbob(
        SPSO,
        {"n_individuals": n},
        name="SPSO",
        evaluations_multiplier=multiplier,
        train=False,
    )"""
    cocopp.main("exdata/DAS_test-0008")
    # cocopp.main("exdata/LMCMAES-0001")
    # cocopp.main("exdata/G3PCX-0001")
    # cocopp.main("exdata/SPSO-0001")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
