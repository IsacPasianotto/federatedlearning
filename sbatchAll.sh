#!/bin/bash
#SBATCH -p DGX
# #SBATCH --account=lade
#SBATCH --job-name=pytorch_distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480GB
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

source $(pwd)/federatedenv/bin/activate

sims=("balanced_center-balanced_proportions" "balanced_center-unbalanced_proportions" "unbalanced_center-balanced_proportions" "unbalanced_center-unbalanced_proportions")

# Set the master node's address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8765
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / $SLURM_GPUS_ON_NODE))

# Preamble to distinguish the jobs
echo "****************************************"
echo "MASTER_ADDR:     $MASTER_ADDR"
echo "DATE:            $(date)"
echo "****************************************"

for sim in ${sims[@]}; 
do
    path=results/$sim
    cp $path/used_settings.py ./settings.py
    RESULTS=$(grep RESULTS_PATH settings.py | awk -F '=' '{print $2}' | tr -d " '")
    mkdir -p $RESULTS
    
    #python3 src/build_local_center_dataset.py

    N_ITER_FED=$(grep N_ITER_FED settings.py | awk -F '=' '{print $2}' | tr -d ' ')
    echo "DOING $sim with $N_ITER_FED iterations"
    
    
    for ((i=0; i<N_ITER_FED; i++))
    do 
        echo "Starting iteration $i"
    
        srun --cpu-bind=none torchrun --nnodes=$SLURM_NNODES \
            --nproc_per_node=$SLURM_GPUS_ON_NODE \
            --rdzv_id="${SLURM_JOB_ID}_${i}" \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            main.py
        wait
        python3 src/aggregate_weights.py
    
        echo "#####################################################"
        echo "     Completed federated learning epoch $((i+1))    "
        echo "#####################################################"
    
    done
done

echo "DONE"
