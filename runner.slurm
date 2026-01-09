#!/bin/bash
#SBATCH --job-name=rl_das_experiment
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgautopt26-gpu-a100

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"

source "$ENV_PATH"

# Run Experiment
python3 dynamicalgorithmselection/main.py example -p 'LMCMAES' 'SPSO' 'G3PCX' --wandb_project RL-DAS --wandb_entity niecwladek-agh --mode LOIO -s 10
