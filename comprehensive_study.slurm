#!/bin/bash
#SBATCH --job-name=rl_das_experiment
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgrldas2026-gpu-a100

# 1st argument: SEED (Default: 42)
SEED=${1:-42}

# Fixed PORTFOLIO variable
PORTFOLIO=('JDE21' 'MADDE' 'NL_SHADE_RSP')

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

echo "Running Mode: Baselines"
python3 dynamicalgorithmselection/main.py BASELINES \
  -p "${PORTFOLIO[@]}" --mode baselines -S "$SEED"
