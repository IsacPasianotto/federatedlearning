########
## Imports
########

# Installed modules:
import os
import sys
import copy
import torch as th

# Defined modules:
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from settings import *

########
## Main function
########

def main() -> None:

    printd("----------------------------")
    printd("Computing aggregation")
    printd("----------------------------")

    net_weights:    list[dict[str, th.Tensor]] = [th.load(f"{RESULTS_PATH}/weights_{i}.pt") for i in range(N_CENTERS)]
    train_sizes:    list[int] = [sum(int(CLASS_SIZES[i] * PERC[i][j] * TRAIN_SIZE) for i in range(N_CLASSES)) for j in range(N_CENTERS)]
    total:          int       = sum(train_sizes)
    center_weights: th.Tensor = (th.tensor(train_sizes, dtype=th.float) / total)

    # move all to cpu becanus they could be in different GPUs
    net_weights = [{ a:b.cpu() for a,b in w.items() } for w in net_weights]

    printd("---------------------------------------")
    printd("Trainsizes for aggregated:", train_sizes)
    printd("---------------------------------------")

    # Perform a deep copy to retrieve just the structure, not the original values
    new_weights: dict[str, th.Tensor] = copy.deepcopy(net_weights[0])

    for k in new_weights.keys():
        stacked_tensors: th.Tensor = th.stack([state_dict[k] for state_dict in net_weights])
        new_weights[k] = (stacked_tensors * center_weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)

    printd("---------------------------------------")
    for i in range(N_CENTERS):
        printd("Saving in",  f"{RESULTS_PATH}/weights_{i}.pt")
        th.save(new_weights, f"{RESULTS_PATH}/weights_{i}.pt")
    printd("---------------------------------------")

if __name__ == '__main__':
    main()
