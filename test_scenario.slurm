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

SEED=123
CDB_VAL=1.5

PORTFOLIO=('G3PCX, CMAES, MADDE')
PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

DIM=3
MODE="CV-LOPO"

echo "Running Mode: $MODE | Dimension: $DIM | CDB: $CDB_VAL | SEED: $SEED"

python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_${MODE}_CDB${CDB_VAL}_DIM${DIM}_SEED${SEED} \
  -p "${PORTFOLIO[@]}" --mode $MODE --dimensionality $DIM \
  --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient --seed $SEED