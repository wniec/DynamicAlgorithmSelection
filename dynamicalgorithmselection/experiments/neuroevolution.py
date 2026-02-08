import multiprocessing
import os
import pickle
import random
import re
from itertools import cycle
from typing import Type

import cocoex
import neat
import numpy as np
import wandb
from tqdm import tqdm

from dynamicalgorithmselection.agents.agent_state import BASE_STATE_SIZE
from dynamicalgorithmselection.experiments.utils import (
    coco_bbob_single_function,
    batched,
    get_suite,
)
from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


def _eval_wrapper(eval_function, seed, genome, config):
    """Wrapper that sets deterministic per-genome seed before evaluation."""
    if seed is not None:
        random.seed(seed + genome.key)
    return eval_function(genome, config)


class ParallelEvaluator:
    def __init__(
        self,
        num_workers,
        eval_function,
        timeout=None,
        initializer=None,
        initargs=(),
        maxtasksperchild=None,
        seed=None,
    ):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.seed = seed
        self.initializer = initializer
        self.initargs = initargs
        self.maxtasksperchild = maxtasksperchild
        self.pool = multiprocessing.Pool(
            processes=num_workers,
            maxtasksperchild=maxtasksperchild,
            initializer=initializer,
            initargs=initargs,
        )
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self.pool is not None and not self._closed:
            self._closed = True
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close()

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            if self.seed is not None:
                jobs.append(
                    self.pool.apply_async(
                        _eval_wrapper, (self.eval_function, self.seed, genome, config)
                    )
                )
            else:
                jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        for job, (ignored_genome_id, genome) in tqdm(
            zip(jobs, genomes),
            total=len(jobs),
            desc="Evaluating Genomes",
            smoothing=0.0,
        ):
            # Result is now a tuple: (fitness, log_dict)
            fitness, log_data = job.get(timeout=self.timeout)
            genome.fitness = fitness

            # Log to wandb in the main process
            if log_data and wandb.run is not None:
                wandb.log(log_data)


class GenomeEvaluator:
    """
    Functor class to evaluate a single genome.
    Allows pickling of configuration/options for multiprocessing.
    """

    def __init__(self, optimizer, options, evaluations_multiplier):
        self.optimizer = optimizer
        self.evaluations_multiplier = evaluations_multiplier
        # Create a clean copy of options (remove unpicklable loggers)
        self.options = options.copy()
        if "run" in self.options:
            del self.options["run"]

    def __call__(self, genome, config):
        # Re-initialize suite inside worker to avoid pickling pointers
        cocoex.utilities.MiniPrint()
        suite = cocoex.Suite("bbob", "", "")

        if not hasattr(genome, "problem_batch"):
            return 0.0, {}

        fitness = 0
        actions = []
        num_actions = len(self.options.get("action_space"))
        local_options = self.options.copy()

        for problem_id in genome.problem_batch:
            problem_instance = suite.get_problem(problem_id)
            local_options["max_function_evaluations"] = (
                self.evaluations_multiplier * problem_instance.dimension
            )
            local_options["train_mode"] = True
            local_options["verbose"] = False
            local_options["net"] = neat.nn.FeedForwardNetwork.create(genome, config)

            results = coco_bbob_single_function(
                self.optimizer, problem_instance, local_options
            )

            fitness += results["mean_reward"]
            local_options["reward_normalizer"] = results["reward_normalizer"]
            actions.extend(results["actions"])

            problem_instance.free()

        # Calculate Entropy
        choices_count = np.array(
            [
                sum(1 for i in actions if i == j) / (len(actions) or 1)
                for j in range(len(actions))
            ]
        )
        norm_entropy = -(
            choices_count
            * np.nan_to_num(np.log(choices_count), nan=0, neginf=0, posinf=0)
        ).sum() / np.log(num_actions)

        final_fitness = fitness / len(genome.problem_batch) + norm_entropy / 35

        # Return fitness AND data to log
        log_data = {
            "fitness": final_fitness,
            "entropy": norm_entropy,
            "mean_reward": fitness / len(genome.problem_batch),
        }
        return final_fitness, log_data


def _coco_bbob_neuroevolution_train(
    optimizer: Type[Optimizer],
    options: dict,
    evaluations_multiplier: int = 1_000,
    mode: str = "easy",
):
    cocoex.utilities.MiniPrint()
    _, problem_ids = get_suite(mode, True)
    batch_size = 30
    input_dim = None
    if options.get("state_representation") == "ELA":
        input_dim = 80
    elif options.get("state_representation") == "NeurELA":
        input_dim = 34
    elif options.get("state_representation") == "custom":
        input_dim = BASE_STATE_SIZE + 2 * len(options.get("action_space")) + 2

    adjust_config(
        input_dim,
        len(options.get("action_space")),
    )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neuroevolution_config",
    )

    population = neat.Population(config)

    all_problem_batches = list(
        batched((np.random.permutation(problem_ids)), batch_size)
    )
    batch_cycler = cycle(all_problem_batches)

    # Pre-assign batches
    for genome_id, genome in population.population.items():
        genome.problem_batch = next(batch_cycler)

    num_workers = len(os.sched_getaffinity(0)) - 1
    print(f"Using {num_workers} workers.")

    evaluator_func = GenomeEvaluator(optimizer, options, evaluations_multiplier)

    with ParallelEvaluator(num_workers, evaluator_func) as pe:
        winner = population.run(pe.evaluate, 80)

    with open(os.path.join("models", f"{options.get('name')}.pkl"), "wb") as f:
        pickle.dump(winner, f)


def adjust_config(n_inputs, n_outputs):
    with open("neuroevolution_config", "r") as f:
        config_content = f.read()

    config_content = re.sub(
        pattern="num_inputs.*", repl=f"num_inputs = {n_inputs}", string=config_content
    )
    config_content = re.sub(
        pattern="num_outputs.*",
        repl=f"num_outputs = {n_outputs}",
        string=config_content,
    )

    with open("neuroevolution_config", "w") as f:
        f.write(config_content)
