########
## Imports
########

# Downloaded Modules
import os
import sys
import torch as th
import torch.distributed as dist

# Defined Modules
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(os.path.join(os.path.dirname(__file__), './modules'))

from dataset import *
from networks import BrainClassifier, printParams
from traintest import train, test
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

    data_file:    str = f"{DATA_PATH}/center_{global_rank}.pt"
    results_file: str = f"{RESULTS_PATH}/GPU_{local_rank}_Node_{node_rank}.pt"

    data: Dataset = th.load(data_file)
    printd(f"Rank {global_rank} loaded {data_file} having {len(data)} images")

    model: BrainClassifier = BrainClassifier().to(device)

    if os.path.isfile(results_file):

        printd(f"Found data, importing from {results_file}")
        net_weights: th.Tensor = th.load(results_file)
        model.load_state_dict(net_weights)

        # Test the model
        acc: float = test(model, device, build_Dataloader(data, BATCH_SIZE))
        printv(f'Accuracy of the aggregated model on center {global_rank}: {acc:.2f}%')


    train_loader, val_loader, test_loader = buildDataloaders(data)
    train_loader: th.utils.data.DataLoader
    val_loader:   th.utils.data.DataLoader
    test_loader:  th.utils.data.DataLoader

    printd("Training on", device, "for", global_rank, "with", len(train_loader), len(train_loader.dataset), "and", len(val_loader), len(val_loader.dataset))

    net_weights: dict      = train(model, device, train_loader, val_loader) 
    acc:         float     = test(model, device, test_loader)
    printv(f"Accuracy of the final model {global_rank} from {device}, node {node_rank} on its test set: {acc:.2f}%")

    # Save the results
    th.save(net_weights, results_file)
    printd(f"Process {global_rank} (Node {node_rank}, GPU {local_rank}) finished training")

    if PRINTWEIGHTS:
        printParams(model)

    # needed sync to avoid hanging processes
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

