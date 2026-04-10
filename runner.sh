#!/bin/bash

SEEDS=(12 23 34)

CDB_VALUES=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1)

PORTFOLIOS=(
    "G3PCX LMCMAES SPSO"
    "MADDE JDE21 NL_SHADE_RSP"
)

echo "Starting job submissions..."

for SEED in "${SEEDS[@]}"; do
    for PORTFOLIO in "${PORTFOLIOS[@]}"; do

        echo "Submitting portfolio study with: SEED=${SEED} | PORTFOLIO=${PORTFOLIO}"

        sbatch portfolio_study.slurm $SEED $PORTFOLIO

        sleep 1

    done

    for CDB in "${CDB_VALUES[@]}"; do

        echo "Submitting CDB study with: SEED=${SEED} | CDB=${CDB}"
        sbatch CDB_study.slurm $SEED $CDB

        sleep 1

    done

    echo "Submitting comprehensive study with: SEED=${SEED}"
    sbatch comprehensive_study.slurm $SEED

    sleep 1

    echo "Submitting reward study with: SEED=${SEED}"
    sbatch reward_study.slurm $SEED

    sleep 1

done

echo "Submitting dummy case study with: SEED=${SEED}"
sbatch dummy_case.slurm $SEED

sleep 1

echo "All jobs submitted!"
