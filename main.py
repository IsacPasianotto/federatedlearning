####
##  Imports
####

# Downloaded Modules
import torch as th
import torch.multiprocessing as mp
import torch.distributed as dist
import os

# Defined Modules
from modules.dataset import *
import modules.networks as networks
from modules.traintest import train, test
from modules.federated import evalAggregate
from settings import *
    
####
## Define main function
####

def main():
    printd("start training")
    dist.init_process_group("nccl")
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    node_rank = global_rank // local_world_size
    num_nodes = world_size // local_world_size

    if th.cuda.is_available():
        device = th.device(f"cuda:{local_rank}")
        th.cuda.set_device(device)
        print(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) loading data")
    else:
        raise RuntimeError("No CUDA devices found. This script requires GPU support. Aborting")

    myIDS = f"GPU{local_rank}_Node{node_rank}"
    data_file = f"{DATA_PATH}/center{global_rank}.pt"
    results_file = f"{RESULTS_PATH}/{myIDS}.pt"
    data = th.load(data_file)
    print(f"Rank {global_rank} loaded {data_file} with {len(data)} images")
    model = networks.BrainClassifier().to(device)
    if os.path.isfile(results_file):
        printd(f"Found data, importing from {results_file}")
        results = th.load(results_file)
        model.load_state_dict(results)
        test(model, device, build_Dataloader(data, BATCH_SIZE), f'Accuracy of the aggregated model on center {global_rank}: ')
    
    # Generate the datasets
    train_loader, val_loader, test_loader = buildDataloaders(data)
    
    results = train(model, device, global_rank, train_loader, val_loader, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
    test(model, device, test_loader, f"Accuracy of the final model {global_rank} from {device}, node {node_rank} on its test set: ")

    # Save the results
    th.save(results, results_file)
    print(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) finished training")

    
    networks.printdParams(model) # Print the number of parameters of the model
    
    dist.destroy_process_group()

if __name__ == '__main__':
    # model = networks.BrainClassifier()
    # total_params = sum(p.numel() for p in model.parameters())
    # printd(f"Total number of parameters: {total_params}")
    main()
