import argparse
import os
import pickle
import shutil
from typing import List, Type
import cocopp
import torch
import wandb

from dynamicalgorithmselection.agents.neuroevolution_agent import NeuroevolutionAgent
from dynamicalgorithmselection.agents.policy_gradient_agent import PolicyGradientAgent
from dynamicalgorithmselection.agents.random_agent import RandomAgent
from dynamicalgorithmselection.experiment import coco_bbob_experiment
from dynamicalgorithmselection import optimizers
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer

AGENTS_DICT = {
    "random": RandomAgent,
    "neuroevolution": NeuroevolutionAgent,
    "policy-gradient": PolicyGradientAgent,
}


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

    parser.add_argument(
        "-a",
        "--agent",
        default="policy-gradient",
        choices=["random", "neuroevolution", "policy-gradient"],
        help="specify which agent to use",
    )

    parser.add_argument(
        "-l",
        "--mode",
        default="easy",
        choices=["LOIO", "hard", "easy"],
        help="specify which agent to use",
    )

    return parser.parse_args()


def print_info(args):
    print("Running an experiment with the following arguments:")

    print("Experiment name: ", args.name)
    print("Portfolio: ", args.portfolio)
    print("Population size: ", args.population_size)
    print("Function eval multiplier: ", args.fe_multiplier)
    print("Test mode: ", args.test)
    print("Compare mode: ", args.compare)
    print("Weights and Biases entity: ", args.wandb_entity)
    print("Weights and Biases project: ", args.wandb_project)


def test(args, action_space):
    if os.path.exists(os.path.join("exdata", f"DAS_test_{args.name}")):
        shutil.rmtree(os.path.join("exdata", f"DAS_test_{args.name}"))

    options = {
        "sub_optimization_ratio": args.sub_optimization_ratio,
        "n_individuals": args.population_size,
        "action_space": action_space,
    }
    # agent_state = torch.load(f)
    if args.agent == "neuroevolution":
        with open(f"DAS_train_{args.name}.pkl", "rb") as f:
            net = pickle.load(f)
        options.update({"net": net})
    elif args.agent == "policy-gradient":
        options.update(torch.load(f"DAS_train_{args.name}.pth", weights_only=False))
    coco_bbob_experiment(AGENTS_DICT[args.agent], options, name=f"DAS_test_{args.name}",
                         evaluations_multiplier=args.fe_multiplier, train=False)
    cocopp.main(os.path.join("exdata", f"DAS_test_{args.name}"))


def main():
    args = parse_arguments()
    print_info(args)
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
    coco_bbob_experiment(AGENTS_DICT[args.agent], {
        "sub_optimization_ratio": args.sub_optimization_ratio,
        "n_individuals": args.population_size,
        "run": run,
        "action_space": action_space,
    }, name=f"DAS_train_{args.name}", evaluations_multiplier=args.fe_multiplier, train=True,
                         agent=args.agent, mode=args.mode)
    if run is not None:
        run.finish()
    if args.test:
        test(args, action_space)
    if args.compare:
        for optimizer in action_space:
            if os.path.exists(os.path.join("exdata", optimizer.__name__)):
                shutil.rmtree(os.path.join("exdata", optimizer.__name__))
            coco_bbob_experiment(optimizer, {"n_individuals": args.population_size}, name=optimizer.__name__,
                                 evaluations_multiplier=args.fe_multiplier, train=False,
                                 agent=None)
            cocopp.main(os.path.join("exdata", optimizer.__name__))


if __name__ == "__main__":
    main()
