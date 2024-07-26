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

    if len(sys.argv) != 3:
        print("Usage: python aggregate_weights.py <nGPUs> <nNodes>")
        sys.exit(1)

    nGPUs:          int = int(sys.argv[1])
    nNodes:         int = int(sys.argv[2])
    nCenters:       int = nGPUs * nNodes
    net_weights:    list[dict[str, th.Tensor]] = [th.load(f"results/weights_{i}.pt") for i in range(nCenters)]
    trainSizes:     list[int]                  = [sum(int(CLASS_SIZES[i] * PERC[i][j] * TRAINSIZE) for i in range(NCLASSES)) for j in range(nCenters)]
    total:          int = sum(trainSizes)
    center_weights: th.Tensor                  = (th.tensor(trainSizes, dtype=th.float) / total)

    # move all to cpu becanus they could be in different GPUs
    net_weights = [{ a:b.cpu() for a,b in w.items() } for w in net_weights]

    printd("---------------------------------------")
    printd("Trainsizes for aggregated:", trainSizes)
    printd("---------------------------------------")

    # Perform a deep copy to retrieve just the structure, not the original values
    new_weights: dict[str, th.Tensor] = copy.deepcopy(net_weights[0])

    for k in new_weights.keys():
        stacked_tensors: th.Tensor = th.stack([state_dict[k] for state_dict in net_weights])
        new_weights[k] = (stacked_tensors * center_weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)

    printd("---------------------------------------")
    for i in range(nCenters):
        printd("Saving in", f"results/weights_{i}.pt")
        th.save(new_weights, f"results/weights_{i}.pt")
    printd("---------------------------------------")

if __name__ == '__main__':
    main()
