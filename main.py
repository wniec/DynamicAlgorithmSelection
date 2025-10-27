import argparse
import os
import shutil
from typing import List, Type
import cocopp
import wandb

from agent import Agent
from experiment import coco_bbob
import optimizers
from optimizers.Optimizer import Optimizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic algorithm selection")

    parser.add_argument(
        "name", type=str, help="Name tag (mandatory positional argument)"
    )

    parser.add_argument(
        "-p",
        "--portfolio",
        nargs="+",
        default=["SPSO", "IPSO", "SPSOL"],
        help="List of portfolio algorithms (default: SPSO IPSO SPSOL)",
    )

    parser.add_argument(
        "-m",
        "--population_size",
        type=int,
        default=20,
        help="Population size (default: 20)",
    )
    parser.add_argument(
        "-s",
        "--sub_optimization_ratio",
        type=int,
        default=10,
        help="ratio of max FE that each sub optimization episode recieves",
    )

    parser.add_argument(
        "-f",
        "--fe_multiplier",
        type=int,
        default=10_000,
        help="Function evaluation multiplier (default: 10 000)",
    )

    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run also in test mode (default: True)",
    )

    parser.add_argument(
        "-c",
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable comparison with each algorithm alone (False by default)",
    )

    parser.add_argument(
        "-e",
        "--wandb_entity",
        type=str,
        default=None,
        help="Enable comparison with each algorithm alone (False by default)",
    )

    parser.add_argument(
        "-w",
        "--wandb_project",
        type=str,
        default=None,
        help="Enable comparison with each algorithm alone (False by default)",
    )

    return parser.parse_args()


def main(args):
    available_optimizers = optimizers.available_optimizers
    action_space: List[Type[Optimizer]] = []
    for optimizer in args.portfolio:
        if optimizer not in available_optimizers:
            raise ValueError(f'Unknown optimizer "{optimizer}"')
        else:
            action_space.append(available_optimizers[optimizer])
    run = None
    if args.wandb_entity is not None and args.wandb_project is not None:
        run = wandb.init(
            name=args.name,
            entity=args.wandb_entity,
            project=args.wandb_project,
            config={
                "dataset": "COCO-BBOB",
            },
        )
    coco_bbob(
        Agent,
        {
            "sub_optimization_ratio": args.sub_optimization_ratio,
            "n_individuals": args.population_size,
            "run": run,
            "action_space": action_space,
        },
        name=f"DAS_train_{args.name}",
        evaluations_multiplier=args.fe_multiplier,
        train=True,
    )
    if run is not None:
        run.finish()
    if args.test:
        if os.path.exists(os.path.join("exdata", f"DAS_test_{args.name}")):
            shutil.rmtree(os.path.join("exdata", f"DAS_test_{args.name}"))
        coco_bbob(
            Agent,
            {
                "sub_optimization_ratio": args.sub_optimization_ratio,
                "n_individuals": args.population_size,
                "action_space": action_space,
            },
            name=f"DAS_test_{args.name}",
            evaluations_multiplier=args.fe_multiplier,
            train=False,
        )
        cocopp.main(os.path.join("exdata", f"DAS_test_{args.name}"))
    if args.compare:
        for optimizer in action_space:
            if os.path.exists(os.path.join("exdata", optimizer.__name__)):
                shutil.rmtree(os.path.join("exdata", optimizer.__name__))
            coco_bbob(
                optimizer,
                {"n_individuals": args.population_size},
                name=optimizer.__name__,
                evaluations_multiplier=args.multiplier,
                train=False,
            )
            cocopp.main(os.path.join("exdata", optimizer.__name__))


if __name__ == "__main__":
    args = parse_arguments()
    print("Running an experiment with the following arguments:")

    print("Experiment name: ", args.name)
    print("Portfolio: ", args.portfolio)
    print("Population size: ", args.population_size)
    print("Function eval multiplier: ", args.fe_multiplier)
    print("Test mode: ", args.test)
    print("Compare mode: ", args.compare)
    print("Weights and Biases entity: ", args.wandb_entity)
    print("Weights and Biases project: ", args.wandb_project)
    main(args)
