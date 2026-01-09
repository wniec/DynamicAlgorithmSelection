import os
from itertools import product
from typing import Type

import cocoex
import numpy as np

from dynamicalgorithmselection.experiments.core import run_testing, run_training
from dynamicalgorithmselection.experiments.utils import (
    ALL_FUNCTIONS,
    INSTANCE_IDS,
    DIMENSIONS,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


def run_cross_validation(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    is_loio: bool = True,
):
    results_dir = os.path.join("results", f"{options.get('name')}")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    cocoex.utilities.MiniPrint()
    problems_suite, cv_folds = _get_cv_folds(4, is_loio)
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
            }
        )
    return observer.result_folder


def _get_cv_folds(n: int, is_loio: bool):
    """
    :param n:  number of cross validation folds
    :param is_loio: boolean to indicate how train and test sets should be split (leave-instance-out/leave-problem-out).
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
    remaining_function_ids = {i for i in ALL_FUNCTIONS}
    test_sets = []
    for i in range(n):
        if is_loio:
            selected = np.random.choice(
                list(remaining_problem_ids),
                size=len(all_problem_ids) // n,
                replace=False,
            ).tolist()
            remaining_problem_ids = remaining_problem_ids.difference(selected)
        else:
            selected_functions = np.random.choice(
                list(remaining_function_ids),
                size=len(ALL_FUNCTIONS) // n,
                replace=False,
            ).tolist()
            selected = [
                i
                for i in all_problem_ids
                if any(i.startswith(f"bbob_f{f_id:03d}") for f_id in selected_functions)
            ]
            remaining_function_ids = remaining_function_ids.difference(
                selected_functions
            )
        test_sets.append(selected)

    return problems_suite, [
        (list(set(all_problem_ids).difference(test_set)), test_set)
        for test_set in test_sets
    ]
