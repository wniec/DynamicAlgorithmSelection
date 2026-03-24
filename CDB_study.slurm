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
#SBATCH --array=0-11           # 12 tasks total

SEED=${1:-42}
CDB_VAL=${2:-1.5}

PORTFOLIO=('G3PCX' 'LMCMAES' 'SPSO')
PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

# Array of Dimensions
DIMS=(2 3 5 10)

echo "Initializing Array Job $SLURM_ARRAY_TASK_ID | SEED: $SEED | CDB: $CDB_VAL | PORTFOLIO: ${PORTFOLIO[*]}"

# 1. Dimension-specific CV-LOIO (Indices 0-3)
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 3 ]]; then
    MODE="CV-LOIO"
    DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
    echo "Running Mode: $MODE | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_${MODE}_CDB${CDB_VAL}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient -S $SEED

# 2. Dimension-specific CV-LOPO (Indices 4-7)
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    MODE="CV-LOPO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "Running Mode: $MODE | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_${MODE}_CDB${CDB_VAL}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient -S $SEED

# 3. Multidimensional CV-LOIO (Index 8)
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then
    MODE="CV-LOIO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_MULTIDIMENSIONAL_${MODE}_CDB${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --cdb $CDB_VAL --agent policy-gradient -S $SEED

# 4. Multidimensional CV-LOPO (Index 9)
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then
    MODE="CV-LOPO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_MULTIDIMENSIONAL_${MODE}_CDB${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --cdb $CDB_VAL --agent policy-gradient -S $SEED

# 5. Multidimensional CV-LODO (Index 10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 10 ]]; then
    MODE="CV-LODO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_PG_MULTIDIMENSIONAL_${MODE}_CDB${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --cdb $CDB_VAL --agent policy-gradient -S $SEED

# 6. Global Random Agent (Index 11)
elif [[ $SLURM_ARRAY_TASK_ID -eq 11 ]]; then
    echo "Running Mode: Global Random Agent"
    python3 dynamicalgorithmselection/main.py ${PORTFOLIO_STR}_${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --cdb $CDB_VAL --agent random -S "$SEED"

fi