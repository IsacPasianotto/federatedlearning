########
## Imports
########

# Downloaded Modules
import os
import torch as th
import torch.distributed as dist
import datetime

# Defined Modules
from modules.dataset import *
from modules.networks import BrainClassifier, print_params
from modules.traintest import train, test
from settings import *

####
## Define main function
####

def main() -> None:

    printd("===========================")
    printd("Start training")
    printd("===========================")

    dist.init_process_group("nccl")

    global_rank: int = int(os.environ["RANK"])
    local_rank:  int = int(os.environ["LOCAL_RANK"])
    node_rank:   int = global_rank // int(os.environ["LOCAL_WORLD_SIZE"])
 
    if th.cuda.is_available():
        device: th.device = th.device(f"cuda:{local_rank}")
        th.cuda.set_device(device)
        printd(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) loading data")
    else:
        raise RuntimeError("No CUDA devices found. This code requires GPU support. Aborting")

    train_data_file:   str = f"{DATA_PATH}/center_{global_rank}.pt"
    test_data_file:    str = f"{DATA_PATH}/center_{global_rank}_test.pt"
    weights_file:      str = f"{RESULTS_PATH}/weights_{global_rank}.pt"
    train_losses_file: str = f"{RESULTS_PATH}/train_losses_{global_rank}.pt"
    val_losses_file:   str = f"{RESULTS_PATH}/val_losses_{global_rank}.pt"
    testAcc_file:      str = f"{RESULTS_PATH}/test_accuracies_{global_rank}.csv"
    aggrAcc_file:      str = f"{RESULTS_PATH}/aggregated_accuracies_{global_rank}.csv"

    train_data: BrainDataset = th.load(train_data_file)
    test_data:  BrainDataset = th.load(test_data_file)
    printd(f"Rank {global_rank} loaded {train_data_file}, {test_data_file} having {len(train_data)} train images and {len(test_data)} test images")

    model: BrainClassifier = BrainClassifier().to(device)

    train_loader, val_loader = build_Dataloaders(train_data)
    train_loader: th.utils.data.DataLoader
    val_loader:   th.utils.data.DataLoader
    test_loader:  th.utils.data.DataLoader = build_Dataloader(test_data)

    if os.path.isfile(weights_file):
        printd(f"Found data, importing from {weights_file}")
        net_weights: dict[str, th.Tensor] = th.load(weights_file)
        model.load_state_dict(net_weights)

        # Test the model
        acc: float = test(model, device, test_loader)
        printv(f'Accuracy of the aggregated model on center {global_rank}: {acc:.2f}%')
        with open(aggrAcc_file, "a") as f:
            f.write(f"{acc}\n")
            
    printd("Training on", device, "for", global_rank, "with", len(train_loader), len(train_loader.dataset), "and", len(val_loader), len(val_loader.dataset))

    net_weights, train_losses, val_losses = train(model, device, train_loader, val_loader) 
    net_weights:  dict[str, th.Tensor]
    train_losses: th.Tensor
    val_losses:   th.Tensor
    
    update_losses(train_losses_file, train_losses)
    update_losses(val_losses_file, val_losses)
    
    acc: float = test(model, device, test_loader)
    with open(testAcc_file, "a") as f:
        f.write(f"{acc}\n")
    
    th.save(net_weights, weights_file)
    
    printv(f"Accuracy of the final model {global_rank} from {device}, node {node_rank} on its test set: {acc:.2f}%")
    printd(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) finished training")

    if PRINT_WEIGHTS:
        print_params(model)
    dist.destroy_process_group()


def update_losses(losses_file, new_losses):
    if os.path.exists(losses_file):
        val_losses_tensor: th.Tensor = th.load(losses_file)
        new_losses = th.cat((val_losses_tensor, new_losses))
    th.save(new_losses, losses_file)

if __name__ == '__main__':
    main()
