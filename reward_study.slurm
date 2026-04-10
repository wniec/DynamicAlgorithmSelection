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
#SBATCH --array=0-3           # 12 tasks total

SEED=${1:-42}
CDB_VAL=1.5

PORTFOLIO=('G3PCX' 'LMCMAES' 'SPSO')
PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

# Array of Dimensions
REWARDS=(1 2 3 4)
DIM=10

echo "Initializing Array Job $SLURM_ARRAY_TASK_ID | SEED: $SEED | CDB: $CDB_VAL | PORTFOLIO: ${PORTFOLIO[*]}"
REWARD=${REWARDS[$SLURM_ARRAY_TASK_ID]}
MODE="CV-LOPO"
echo "Running Mode: $MODE | Dimension: $DIM"

python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_${MODE}_CDB${CDB_VAL}_DIM${DIM}_REWARD${REWARD}_SEED${SEED} \
  -p "${PORTFOLIO[@]}" --mode $MODE --dimensionality $DIM \
  --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient -S $SEED -O $REWARD
