# Downloaded Modules
import torch as th
import torch.distributed as dist
import os

# Defined Modules
from modules.dataset import *
from modules.networks import BrainClassifier, printParams
from modules.traintest import train, test
from modules.settings import *
    
####
## Define main function
####

def main():
    printd("start training")
    dist.init_process_group("nccl")
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    node_rank = global_rank // int(os.environ["LOCAL_WORLD_SIZE"])

    if th.cuda.is_available():
        device = th.device(f"cuda:{local_rank}")
        th.cuda.set_device(device)
        printd(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) loading data")
    else:
        raise RuntimeError("No CUDA devices found. This script requires GPU support. Aborting")

    data_file = f"{DATA_PATH}/center{global_rank}.pt"
    data = th.load(data_file)
    printd(f"Rank {global_rank} loaded {data_file} having {len(data)} images")
    model = BrainClassifier().to(device)
    results_file = f"{RESULTS_PATH}/GPU{local_rank}_Node{node_rank}.pt"
    if os.path.isfile(results_file):
        printd(f"Found data, importing from {results_file}")
        net_weights = th.load(results_file)
        model.load_state_dict(net_weights)
        acc = test(model, device, build_Dataloader(data, BATCH_SIZE))
        printv(f'Accuracy of the aggregated model on center {global_rank}: {acc:.2f}%')
    
    train_loader, val_loader, test_loader = buildDataloaders(data)
    printd("Training on", device, " for ", global_rank, " with ", len(train_loader), len(train_loader.dataset), " and ", len(val_loader), len(val_loader.dataset))
    net_weights = train(model, device, train_loader, val_loader)
    acc = test(model, device, test_loader)
    printv(f"Accuracy of the final model {global_rank} from {device}, node {node_rank} on its test set: {acc:.2f}%")

    # Save the results
    th.save(net_weights, results_file)
    printd(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) finished training")

    if PRINTWEIGHTS:
        printParams(model) # Print the number of parameters of the model
    dist.destroy_process_group()

if __name__ == '__main__':
    # model = networks.BrainClassifier()
    # total_params = sum(p.numel() for p in model.parameters())
    # printd(f"Total number of parameters: {total_params}")
    main()
