import json

import cocoex
import cocopp
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from agent import Agent
from optimizers.G3PCX import G3PCX
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


def coco_bbob(
        optimizer: Agent,
        options: dict,
        name: str,
        evaluations_multiplier: int = 1_000,
        train: bool = True,
        easy_mode: bool = True,
):
    all_actor_losses = []
    all_critic_losses = []
    if train:
        actor_params, critic_params, actor_optimizer, critic_optimizer = (
            None,
            None,
            None,
            None,
        )

    else:
        state_dict = torch.load("state.pth")
        actor_params = state_dict["actor_params"]
        critic_params = state_dict["critic_params"]
        actor_optimizer = state_dict["actor_optimizer"]
        critic_optimizer = state_dict["critic_optimizer"]
    suite, output = "bbob", name
    observer = cocoex.Observer(suite, "result_folder: " + output)
    cocoex.utilities.MiniPrint()
    working_set = (
        EASY_TRAIN_BBOB
        if ((easy_mode and train) or (not easy_mode and not train))
        else ALL_FUNCTIONS.difference(EASY_TRAIN_BBOB)
    )
    working_set_filter = lambda f: f.id_function in working_set
    for function in tqdm(filter(working_set_filter, cocoex.Suite(suite, "", ""))):
        function.observe_with(observer)
        options["max_function_evaluations"] = (
                evaluations_multiplier * function.dimension
        )
        options["actor_params"] = actor_params
        options["critic_params"] = critic_params
        options["actor_optimizer"] = actor_optimizer
        options["critic_optimizer"] = critic_optimizer
        options["train"] = train
        options["verbose"] = False
        if train:
            results, (actor_params, critic_params, actor_optimizer, critic_optimizer) = (
                coco_bbob_single_function(optimizer, function, options)
            )
            all_actor_losses.extend(results["actor_losses"])
            all_critic_losses.extend(results["critic_losses"])
        else:
            coco_bbob_single_function(optimizer, function, options)
    if train:
        state_dict = {"actor_params": actor_params,
                      "critic_params": critic_params,
                      "actor_optimizer": actor_optimizer,
                      "critic_optimizer": critic_optimizer}
        torch.save(state_dict, "state.pth")
        with open("actor_losses.json", "w") as file:
            json.dump(all_actor_losses, file)
        with open("critic_losses.json", "w") as file:
            json.dump(all_critic_losses, file)
    return observer.result_folder


def coco_bbob_single_function(optimizer, function: cocoex.interface.Problem, options):
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
    coco_bbob(
        LMCMAES,
        {"n_individuals": n},
        name="LMCMAES",
        evaluations_multiplier=multiplier,
        train=False,
    )
    coco_bbob(
        G3PCX,
        {"n_individuals": n},
        name="G3PCX",
        evaluations_multiplier=multiplier,
        train=False,
    )
    coco_bbob(
        SPSO,
        {"n_individuals": n},
        name="SPSO",
        evaluations_multiplier=multiplier,
        train=False,
    )
    cocopp.main("exdata/DAS_test")
    cocopp.main("exdata/LMCMAES")
    cocopp.main("exdata/G3PCX")
    cocopp.main("exdata/SPSO")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
