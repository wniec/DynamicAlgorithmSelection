#!/bin/bash
#SBATCH --job-name=rl_das_experiment
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-9           # Increased to 24 tasks total to split sequential runs

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

# 2. Dimension-specific CV-LOIO | Policy Gradient (Indices 0-3)
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 3 ]]; then
    MODE="CV-LOIO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID))]}
    echo "Running Mode: $MODE | Agent: Policy Gradient | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${CDB_VAL}_${DIM} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient

# 4. Dimension-specific CV-LOPO | Policy Gradient (Indices 4-7)
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    MODE="CV-LOPO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "Running Mode: $MODE | Agent: Policy Gradient | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${CDB_VAL}_${DIM} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient

# 6. Multidimensional CV-LOIO (Index 20)
elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then
    MODE="CV-LOIO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --cdb $CDB_VAL --agent policy-gradient

# 7. Multidimensional CV-LOPO (Index 21)
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then
    MODE="CV-LOPO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --cdb $CDB_VAL --agent policy-gradient

# 8. Global Random Agent (Index 22)
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then
    echo "Running Mode: Global Random Agent"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM_${CDB_VAL} \
      -p "${PORTFOLIO[@]}" --cdb $CDB_VAL --agent random
fi