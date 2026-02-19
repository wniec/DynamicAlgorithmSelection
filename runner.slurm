#!/bin/bash
#SBATCH --job-name=rl_das_experiment
#SBATCH --output=logs/experiment_%A_%a.out
#SBATCH --error=logs/experiment_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --array=0-15           # 0-6 (LOIO), 7-13 (LOPO), 14-15 (Random and baselines)

CDB_VAL=${1:-1.5}

# CONFIGURATION
ENV_PATH="$SCRATCH/DynamicAlgorithmSelection/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs  # Ensure logs directory exists

Determine the Mode based on the Array ID
if [[ $SLURM_ARRAY_TASK_ID -le 6 || $SLURM_ARRAY_TASK_ID -ge 14 ]]; then
    MODE="CV-LOIO"
    TASK_ID=$SLURM_ARRAY_TASK_ID
else
    MODE="CV-LOPO"
    TASK_ID=$((SLURM_ARRAY_TASK_ID - 7))
fi

# Map the Task ID (0-6) to specific experiments
DIMS=(2 3 5 10 20 40)

if [ $TASK_ID -le 5 ]; then
    # --- DIMENSION SPECIFIC RUNS ---
    DIM=${DIMS[$TASK_ID]}
    echo "Running Mode: $MODE | Dimension: $DIM"

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RLDAS_${MODE}_${DIM} \
      -p 'JDE21' 'MADDE' 'NL_SHADE_RSP'  --mode $MODE --dimensionality $DIM --n_epochs 40 --agent RL-DAS

    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_${MODE}_${DIM} \
      -p 'JDE21' 'MADDE' 'NL_SHADE_RSP' -r custom --mode $MODE --dimensionality $DIM \
      --cdb $CDB_VAL --n_epochs 3 --agent policy-gradient

elif [ $TASK_ID -eq 6 ]; then
    # --- MULTIDIMENSIONAL ---
    echo "Running Mode: $MODE | Multidimensional PG"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_PG_MULTIDIMENSIONAL_$MODE \
      -p 'JDE21' 'MADDE' 'NL_SHADE_RSP' -r custom --mode $MODE --cdb $CDB_VAL --agent policy-gradient

elif [ $TASK_ID -eq 14 ]; then
    # --- RANDOM AGENT ---
    echo "Running Mode: Random Agent"
    python3 dynamicalgorithmselection/main.py JDE21_MADDE_NL_SHADE_RSP_RANDOM \
      -p 'JDE21' 'MADDE' 'NL_SHADE_RSP' --cdb $CDB_VAL --agent random

elif [ $TASK_ID -eq 15 ]; then
    # --- BASELINES ---
    echo "Running Mode: Baselines"
    python3 dynamicalgorithmselection/main.py BASELINES \
      -p 'JDE21' 'MADDE' 'NL_SHADE_RSP' --agent random --mode baselines
fi