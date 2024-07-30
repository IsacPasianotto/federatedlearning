#!/bin/bash
#SBATCH -p DGX
#SBATCH --job-name=pytorch_distributed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

source ../dgx/bin/activate


# Preamble to distinguish the jobs
echo "****************************************"
echo "DATE:            $(date)"
echo "****************************************"

python3 src/baseline.py

echo "############################"
echo "     Completed training     "
echo "############################"
