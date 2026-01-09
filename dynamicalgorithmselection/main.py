import argparse
import os
import pickle
import shutil
from typing import List, Type
import cocopp
import neat
import torch
import wandb

from dynamicalgorithmselection.agents.neuroevolution_agent import NeuroevolutionAgent
from dynamicalgorithmselection.agents.policy_gradient_agent import PolicyGradientAgent
from dynamicalgorithmselection.agents.random_agent import RandomAgent
from dynamicalgorithmselection.experiments.experiment import coco_bbob_experiment
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
        "--n_checkpoints",
        type=int,
        default=10,
        help="number of checkpoints, where sub-optimizer is chosen",
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
        type=str,
        default="policy-gradient",
        choices=["random", "neuroevolution", "policy-gradient"],
        help="specify which agent to use",
    )

    parser.add_argument(
        "-l",
        "--mode",
        type=str,
        default="easy",
        choices=["LOIO", "hard", "easy", "CV-LOIO", "CV-LOPO", "baselines"],
        help="specify which agent to use",
    )

    parser.add_argument(
        "-r",
        "--state-representation",
        type=str,
        default="ELA",
        choices=["ELA", "NeurELA", "custom"],
        help="specify which state representation to use",
    )

    parser.add_argument(
        "-x",
        "--cdb",
        type=float,
        default=1.0,
        help="checkpoint division exponent",
    )

    parser.add_argument(
        "-d",
        "--force-restarts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable selection of forcibly restarting optimizers",
    )
    return parser.parse_args()


def print_info(args):
    if args.agent == "random" and not args.test:
        raise ValueError("Random agent is available for testing only.")

    print("Running an experiment with the following arguments:")

    print("Experiment name: ", args.name)
    print("Portfolio: ", args.portfolio)
    print("Population size: ", args.population_size)
    print("Function eval multiplier: ", args.fe_multiplier)
    print("Test mode: ", args.test)
    print("Compare mode: ", args.compare)
    print("Weights and Biases entity: ", args.wandb_entity)
    print("Weights and Biases project: ", args.wandb_project)
    print("Agent type: ", args.agent if args.mode != "baselines" else None)
    print("Exponential checkpoint division base: ", args.cdb)
    print("State representation variant: ", args.state_representation)
    print("Forcing restarts: ", args.force_restarts)


def test(args, action_space):
    if os.path.exists(os.path.join("exdata", f"DAS_{args.name}")):
        shutil.rmtree(os.path.join("exdata", f"DAS_{args.name}"))

    options = {
        "n_checkpoints": args.n_checkpoints,
        "n_individuals": args.population_size,
        "action_space": action_space,
        "cdb": args.cdb,
        "state_representation": args.state_representation,
        "force_restarts": args.force_restarts,
    }
    # agent_state = torch.load(f)
    if args.agent == "neuroevolution":
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "neuroevolution_config",
        )
        with open(os.path.join("models", f"DAS_train_{args.name}.pkl"), "rb") as f:
            winner_genome = pickle.load(f)
            net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        options.update({"net": net})
    elif args.agent == "policy-gradient":
        options.update(
            torch.load(
                os.path.join("models", f"DAS_train_{args.name}_final.pth"),
                weights_only=False,
            )
        )
    coco_bbob_experiment(
        AGENTS_DICT[args.agent],
        options,
        name=f"DAS_{args.name}",
        evaluations_multiplier=args.fe_multiplier,
        train=False,
        agent=args.agent,
        mode=args.mode,
    )
    cocopp.main(os.path.join("exdata", f"DAS_{args.name}"))


def run_training(args, action_space):
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
    coco_bbob_experiment(
        AGENTS_DICT[args.agent],
        {
            "n_checkpoints": args.n_checkpoints,
            "n_individuals": args.population_size,
            "run": run,
            "action_space": action_space,
            "cdb": args.cdb,
            "state_representation": args.state_representation,
            "force_restarts": args.force_restarts,
        },
        name=f"DAS_train_{args.name}",
        evaluations_multiplier=args.fe_multiplier,
        train=True,
        agent=args.agent,
        mode=args.mode,
    )
    if run is not None:
        run.finish()


def run_CV(args, action_space):
    if os.path.exists(os.path.join("exdata", f"DAS_CV_{args.name}")):
        shutil.rmtree(os.path.join("exdata", f"DAS_CV_{args.name}"))
    coco_bbob_experiment(
        AGENTS_DICT[args.agent],
        {
            "n_checkpoints": args.n_checkpoints,
            "n_individuals": args.population_size,
            "run": None,
            "action_space": action_space,
            "cdb": args.cdb,
            "state_representation": args.state_representation,
            "force_restarts": args.force_restarts,
        },
        name=f"DAS_CV_{args.name}",
        evaluations_multiplier=args.fe_multiplier,
        train=True,
        agent=args.agent,
        mode=args.mode,
    )
    cocopp.main(os.path.join("exdata", f"DAS_CV_{args.name}"))


def run_baselines(args, action_space):
    for optimizer in action_space:
        if os.path.exists(os.path.join("exdata", optimizer.__name__)):
            shutil.rmtree(os.path.join("exdata", optimizer.__name__))
        coco_bbob_experiment(
            None,
            {
                "optimizer_portfolio": action_space,
                "n_individuals": args.population_size,
                "baselines": True,
                "n_checkpoints": args.n_checkpoints,
                "cdb": args.cdb,
                "state_representation": args.state_representation,
                "force_restarts": args.force_restarts,
            },
            name=optimizer.__name__,
            evaluations_multiplier=args.fe_multiplier,
            train=False,
            agent=None,
        )
        cocopp.main(os.path.join("exdata", optimizer.__name__))


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
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists("results"):
        os.mkdir("results")
    if args.mode.startswith("CV"):
        run_CV(args, action_space)
    else:
        if args.agent != "random" and args.mode != "baselines":
            run_training(args, action_space)
        if args.test and args.mode != "baselines":
            test(args, action_space)
    if args.compare or args.mode == "baselines":
        run_baselines(args, action_space)


if __name__ == "__main__":
    main()
