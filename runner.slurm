#!/bin/bash
#SBATCH --job-name=bbo_experiment_array
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=plggautoopt26

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"

# shellcheck disable=SC1090
source "$ENV_PATH"

# Run Experiment
echo "Running experiment $SLURM_ARRAY_TASK_ID"
uv run das loio_simple_fe_multiplication -p 'LMCMAES' 'SPSO' 'G3PCX' --mode LOIO --agent random