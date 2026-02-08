from typing import Type

import cocoex
import numpy as np
from tqdm import tqdm

from dynamicalgorithmselection.experiments.utils import (
    coco_bbob_single_function,
    dump_stats,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


def run_testing(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int,
    problems_suite: cocoex.Suite,
    problem_ids: list[str],
    observer: cocoex.Observer,
):
    for problem_id in tqdm(problem_ids, smoothing=0.0):
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


def run_training(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int,
    problems_suite: cocoex.Suite,
    problem_ids: list[str],
):
    agent_state = {}
    for problem_id in tqdm(np.random.permutation(problem_ids), smoothing=0.0):
        problem_instance = problems_suite.get_problem(problem_id)
        max_fe = evaluations_multiplier * problem_instance.dimension
        options["max_function_evaluations"] = max_fe
        options.update(agent_state)
        options["train_mode"] = True
        options["verbose"] = False
        results, agent_state = coco_bbob_single_function(
            optimizer, problem_instance, options
        )
        options["state_normalizer"] = agent_state["state_normalizer"]
        options["reward_normalizer"] = agent_state["reward_normalizer"]
        options["buffer"] = agent_state["buffer"]
        problem_instance.free()
