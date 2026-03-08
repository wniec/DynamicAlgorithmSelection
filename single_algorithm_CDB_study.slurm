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
#SBATCH --array=0-9           # 10 tasks total

CDB_VAL=1.5

PORTFOLIO=('MADDE')

PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")


# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

# Array of Dimensions
DIMS=(2 3 5 10)

MODE="CV-LOPO"
DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
echo "Running Mode: $MODE | Dimension: $DIM"

python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_${MODE}_CDB${CDB_VAL}_DIM${DIM} \
  -p "${PORTFOLIO[@]}" --mode $MODE --dimensionality $DIM \
  --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient
