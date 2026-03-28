import json
import os
from typing import Type, Optional

from dynamicalgorithmselection.experiments.core import run_testing, run_training
from dynamicalgorithmselection.experiments.cross_validation import run_cross_validation
from dynamicalgorithmselection.experiments.utils import (
    coco_bbob_single_function,
    get_suite,
    dump_stats,
)

import cocoex
from tqdm import tqdm

from dynamicalgorithmselection.agents.agent_utils import (
    get_extreme_stats,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


def dump_extreme_stats(
    name: str,
    stats,
    problem_instance,
    max_function_evaluations,
):
    best_case, worst_case = get_extreme_stats(stats, max_function_evaluations)
    os.makedirs("results", exist_ok=True)
    for suffix, case in [("best", best_case), ("worst", worst_case)]:
        with open(os.path.join("results", f"{name}_{suffix}.jsonl"), "a") as f:
            f.write(json.dumps({problem_instance: case}) + "\n")


def coco_bbob_experiment(
    optimizer: Optional[Type[Optimizer]],
    options: dict,
    name: str,
    evaluations_multiplier: int = 1_000,
    train: bool = True,
    mode: str = "easy",
    agent: Optional[str] = "policy-gradient",
):
    options["name"] = name
    if mode.startswith("CV"):
        return run_cross_validation(
            optimizer=optimizer,
            options=options,
            evaluations_multiplier=evaluations_multiplier,
            leaving_mode=mode[-4:],
        )
    elif agent in ["random", "RL-DAS-random"]:
        return _coco_bbob_test_all(optimizer, options, evaluations_multiplier, mode)
    elif options.get("baselines"):
        return run_comparison(
            options["optimizer_portfolio"], options, evaluations_multiplier
        )
    elif not train:
        return _coco_bbob_test(optimizer, options, evaluations_multiplier, mode)
    else:
        return _coco_bbob_policy_gradient_train(
            optimizer, options, evaluations_multiplier, mode
        )


def _coco_bbob_policy_gradient_train(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    mode: str = "easy",
):
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite(mode, True, options.get("dimensionality"))
    options["n_problems"] = len(problem_ids)
    run_training(
        optimizer, options, evaluations_multiplier, problems_suite, problem_ids
    )


def _coco_bbob_test(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    mode: str = "easy",
):
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite(mode, False, options.get("dimensionality"))
    options["n_problems"] = len(problem_ids)
    observer = cocoex.Observer("bbob", "result_folder: " + options["name"])
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
    cocoex.utilities.MiniPrint()
    problems_suite, problem_ids = get_suite(
        "baselines", False, options.get("dimensionality")
    )
    options["n_problems"] = len(problem_ids)
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


def run_comparison(
    optimizer_portfolio: list[Type[Optimizer]],
    options: dict,
    evaluations_multiplier: int,
):
    observers = {}
    suites = {}
    results_folders = []

    print("Initializing Observers...")
    for optimizer in optimizer_portfolio:
        optimizer_name = optimizer.__name__
        case_name = f"{options['name']}_{optimizer_name}"

        observer = cocoex.Observer("bbob", "result_folder: " + case_name)
        observers[optimizer_name] = observer
        results_folders.append("exdata/" + case_name)  # Adjust path if needed

        suites[optimizer_name] = get_suite("all", False, options.get("dimensionality"))[
            0
        ]

    cocoex.utilities.MiniPrint()

    # We use the problem_ids from the first suite to iterate
    _, problem_ids = get_suite("all", False, options.get("dimensionality"))
    options["n_problems"] = len(problem_ids)

    for problem_id in tqdm(problem_ids, desc="Evaluating Problems", smoothing=0.0):
        stats = {}
        max_fe = None

        for optimizer in optimizer_portfolio:
            optimizer_name = optimizer.__name__
            result_folder_name = f"{options['name']}_{optimizer_name}"

            problem_instance = suites[optimizer_name].get_problem(problem_id)
            problem_instance.observe_with(observers[optimizer_name])

            max_fe = evaluations_multiplier * problem_instance.dimension
            options["max_function_evaluations"] = max_fe
            options["train_mode"] = False
            options["verbose"] = False
            results = coco_bbob_single_function(optimizer, problem_instance, options)

            problem_instance.free()  # Flushes data to disk

            stats[optimizer_name] = results["fitness_history"]
            dump_stats(
                results[0] if isinstance(results, tuple) else results,
                result_folder_name,
                problem_id,
                max_fe,
            )

        dump_extreme_stats(
            options.get("name"),
            stats,
            problem_id,
            max_fe,
        )
