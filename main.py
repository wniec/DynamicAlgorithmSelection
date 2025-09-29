import cocoex
import numpy as np
from pypop7.optimizers.core import Optimizer

from agent import Agent


def coco_bbob(
        optimizer: Optimizer,
        options: dict,
        evaluations_multiplier: int = 1_000,
        run_id: id = 0,
):
    suite, output = "bbob", f"{run_id}_{evaluations_multiplier}"
    observer = cocoex.Observer(suite, "result_folder: " + output)
    cocoex.utilities.MiniPrint()
    for i, function in enumerate(cocoex.Suite(suite, "", "")):
        function.observe_with(observer)
        options["max_function_evaluations"] = (
                evaluations_multiplier * function.dimension
        )
        # options["verbose"] = False
        coco_bbob_single_function(optimizer, function, options)
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


if __name__ == '__main__':
    coco_bbob(
        Agent,
        {'sub_optimization_ratio': 2, 'n_individuals': 20},
        800,
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
