#!/bin/bash
#SBATCH --job-name=rl_das_experiment
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-15           # 16 tasks total

CDB_VAL=${1:-1.5}

if [ "$#" -gt 0 ]; then
    shift
fi

# Store the remaining arguments as an array called PORTFOLIO.
# If no additional arguments were provided, fall back to your default.
if [ "$#" -eq 0 ]; then
    PORTFOLIO=('JDE21' 'MADDE' 'NL_SHADE_RSP')
else
    PORTFOLIO=("$@")
fi

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

# Array of Dimensions
DIMS=(2 3 5 10)

# 1. Dimension-specific CV-LOIO (Indices 0-3)
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 3 ]]; then
    MODE="CV-LOIO"
    DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
    echo "Running Mode: $MODE | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RLDAS_${MODE}_${DIM} \
      -p "${PORTFOLIO[@]}"  --mode $MODE --dimensionality $DIM --n_epochs 40 --agent RL-DAS

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${CDB_VAL}_${DIM} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient

# 2. Dimension-specific CV-LOPO (Indices 4-7)
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    MODE="CV-LOPO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "Running Mode: $MODE | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RLDAS_${MODE}_${DIM} \
      -p "${PORTFOLIO[@]}"  --mode $MODE --dimensionality $DIM --n_epochs 40 --agent RL-DAS

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${CDB_VAL}_${DIM} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient

# 3. Dimension-specific RL-DAS-random (Indices 8-11)
elif [[ $SLURM_ARRAY_TASK_ID -ge 8 && $SLURM_ARRAY_TASK_ID -le 11 ]]; then
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 8))]}
    echo "Running Mode: Random Agent - RLDAS variant | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM_DAS_${DIM} \
      -p 'JDE21' 'MADDE' 'NL_SHADE_RSP' --cdb $CDB_VAL --agent RL-DAS-random --dimensionality $DIM

# 4. Multidimensional CV-LOIO (Index 12)
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]; then
    MODE="CV-LOIO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --cdb $CDB_VAL --agent policy-gradient

# 5. Multidimensional CV-LOPO (Index 13)
elif [[ $SLURM_ARRAY_TASK_ID -eq 13 ]]; then
    MODE="CV-LOPO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --cdb $CDB_VAL --agent policy-gradient

# 6. Global Random Agent (Index 14)
elif [[ $SLURM_ARRAY_TASK_ID -eq 14 ]]; then
    echo "Running Mode: Global Random Agent"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM_${CDB_VAL} \
      -p "${PORTFOLIO[@]}" --cdb $CDB_VAL --agent random

# 7. Global Baselines (Index 15)
elif [[ $SLURM_ARRAY_TASK_ID -eq 15 ]]; then
    echo "Running Mode: Baselines"
    python3 dynamicalgorithmselection/main.py BASELINES \
      -p "${PORTFOLIO[@]}" --agent random --mode baselines
fi