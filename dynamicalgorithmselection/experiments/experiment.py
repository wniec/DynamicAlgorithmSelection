import json
import os
from typing import Type, Optional

from dynamicalgorithmselection.experiments.core import run_testing, run_training
from dynamicalgorithmselection.experiments.cross_validation import run_cross_validation
from dynamicalgorithmselection.experiments.neuroevolution import (
    _coco_bbob_neuroevolution_train,
)
from dynamicalgorithmselection.experiments.utils import (
    coco_bbob_single_function,
    get_suite,
    dump_stats,
)

import cocoex
from tqdm import tqdm

from dynamicalgorithmselection.agents.agent_utils import (
    get_extreme_stats,
    get_checkpoints,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


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
    if mode.startswith("CV"):
        return run_cross_validation(
            optimizer, options, evaluations_multiplier, is_loio=mode.endswith("LOIO")
        )
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
