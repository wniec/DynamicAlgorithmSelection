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
#SBATCH --array=0-12           # Increased to 13 tasks total to split sequential runs

# 1st argument: SEED (Default: 42)
SEED=${1:-42}

# Fixed PORTFOLIO variable
PORTFOLIO=('JDE21' 'MADDE' 'NL_SHADE_RSP')

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

# Array of Dimensions
DIMS=(2 3 5 10)

# 1. Dimension-specific CV-LOIO | RL-DAS (Indices 0-3)
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 3 ]]; then
    MODE="CV-LOIO"
    DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
    echo "Running Mode: $MODE | Agent: RL-DAS | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RLDAS_${MODE}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}"  --mode $MODE --dimensionality $DIM --n_epochs 40 --agent RL-DAS -S "$SEED"

# 3. Dimension-specific CV-LOPO | RL-DAS (Indices 4-7)
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    MODE="CV-LOPO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "Running Mode: $MODE | Agent: RL-DAS | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RLDAS_${MODE}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}"  --mode $MODE --dimensionality $DIM --n_epochs 40 --agent RL-DAS -S "$SEED"

# 5. Dimension-specific RL-DAS-random (Indices 8-11)
elif [[ $SLURM_ARRAY_TASK_ID -ge 8 && $SLURM_ARRAY_TASK_ID -le 11 ]]; then
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 8))]}
    echo "Running Mode: Random Agent - RLDAS variant | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM_DAS_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --agent RL-DAS-random --dimensionality $DIM -S "$SEED"

# 6. Global Baselines (Index 12)
elif [[ $SLURM_ARRAY_TASK_ID -eq 12 ]]; then
    echo "Running Mode: Baselines"
    python3 dynamicalgorithmselection/main.py BASELINES \
      -p "${PORTFOLIO[@]}" --mode baselines -S "$SEED"
fi