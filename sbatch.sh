#!/bin/bash
#SBATCH -p DGX
#SBATCH --job-name=pytorch_distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00

source ../dgx/bin/activate 

rm -rf data/center*
rm -rf results/
mkdir results/

python3 src/build_local_center_dataset.py

# Set the master node's address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))

NITER_FED=$(python3 -c "from settings import NITER_FED; print(NITER_FED)")

for ((i=0; i<NITER_FED; i++))
do 
    echo "Starting iteration $i"
    srun --cpu-bind=none torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id="${SLURM_JOB_ID}_${i}" --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT main.py
    echo "Iteration $i completed"
    python3 src/aggregate_weights.py $SLURM_GPUS_ON_NODE $SLURM_NNODES
    echo "############################# Completed federated learning epoch $((i+1)) #############################"
done

echo "DONE"
