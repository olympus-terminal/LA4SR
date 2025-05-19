#!/bin/bash

#SBATCH -o slurm-logs/arrayJob_%A_%a.out
#SBATCH -e slurm-logs/arrayJob_%A_%a.err
#SBATCH -a 1-12  #5-112  # <-- set to length of the *longer* file
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

# Get line count of each file
NUM_ALGAE=$(wc -l < algae-filelist.txt)
NUM_CONTAM=$(wc -l < contam-filelist.txt)

# Use raw SLURM task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Modulo wrap if needed
IDX_ALGAE=$(( (TASK_ID - 1) % NUM_ALGAE + 1 ))
IDX_CONTAM=$(( (TASK_ID - 1) % NUM_CONTAM + 1 ))

# Extract lines from files
ALINE=$(sed -n "${IDX_ALGAE}p" algae-filelist.txt)
CLINE=$(sed -n "${IDX_CONTAM}p" contam-filelist.txt)

# Run your classification script
./run_la4sr_TI-inc-algaGPT.sh resume "$ALINE" "$CLINE"

## EXAMPLE:

##./run_la4sr.sh ./test-data/TI-free/AlgalTop10000-10holdout-headed.fa ./test-data/TI-free/BactTop10000-10holdout-headed.fa
