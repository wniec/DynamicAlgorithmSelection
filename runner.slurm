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
#SBATCH --array=0-23           # Increased to 24 tasks total to split sequential runs

# 1st argument: CDB_VAL (Default: 1.5)
CDB_VAL=${1:-1.5}

# 2nd argument: SEED (Default: 42)
SEED=${2:-42}

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

# 2. Dimension-specific CV-LOIO | Policy Gradient (Indices 4-7)
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    MODE="CV-LOIO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "Running Mode: $MODE | Agent: Policy Gradient | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${CDB_VAL}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" -r custom --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient -S "$SEED"

# 3. Dimension-specific CV-LOPO | RL-DAS (Indices 8-11)
elif [[ $SLURM_ARRAY_TASK_ID -ge 8 && $SLURM_ARRAY_TASK_ID -le 11 ]]; then
    MODE="CV-LOPO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 8))]}
    echo "Running Mode: $MODE | Agent: RL-DAS | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RLDAS_${MODE}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}"  --mode $MODE --dimensionality $DIM --n_epochs 40 --agent RL-DAS -S "$SEED"

# 4. Dimension-specific CV-LOPO | Policy Gradient (Indices 12-15)
elif [[ $SLURM_ARRAY_TASK_ID -ge 12 && $SLURM_ARRAY_TASK_ID -le 15 ]]; then
    MODE="CV-LOPO"
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 12))]}
    echo "Running Mode: $MODE | Agent: Policy Gradient | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${CDB_VAL}_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient -S "$SEED"

# 5. Dimension-specific RL-DAS-random (Indices 16-19)
elif [[ $SLURM_ARRAY_TASK_ID -ge 16 && $SLURM_ARRAY_TASK_ID -le 19 ]]; then
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 16))]}
    echo "Running Mode: Random Agent - RLDAS variant | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM_DAS_DIM${DIM}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --agent RL-DAS-random --dimensionality $DIM -S "$SEED"

# 6. Multidimensional CV-LOIO (Index 20)
elif [[ $SLURM_ARRAY_TASK_ID -eq 20 ]]; then
    MODE="CV-LOIO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --cdb $CDB_VAL --agent policy-gradient -S "$SEED"

# 7. Multidimensional CV-LOPO (Index 21)
elif [[ $SLURM_ARRAY_TASK_ID -eq 21 ]]; then
    MODE="CV-LOPO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --cdb $CDB_VAL --agent policy-gradient -S "$SEED"

# 8. Global Random Agent (Index 22)
elif [[ $SLURM_ARRAY_TASK_ID -eq 22 ]]; then
    echo "Running Mode: Global Random Agent"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM_${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --cdb $CDB_VAL --agent random -S "$SEED"

# 9. Global Baselines (Index 23)
elif [[ $SLURM_ARRAY_TASK_ID -eq 23 ]]; then
    echo "Running Mode: Baselines"
    python3 dynamicalgorithmselection/main.py BASELINES \
      -p "${PORTFOLIO[@]}" --mode baselines -S "$SEED"

# 10. Multidimensional CV-LODO (Index 24)
elif [[ $SLURM_ARRAY_TASK_ID -eq 24 ]]; then
    MODE="CV-LOPO"
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_${MODE}_${CDB_VAL}_SEED${SEED} \
      -p "${PORTFOLIO[@]}" --mode $MODE --cdb $CDB_VAL --agent policy-gradient -S "$SEED"
fi