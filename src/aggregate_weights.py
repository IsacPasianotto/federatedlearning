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
    center_sizes:   list[int] = [sum(int(class_size*center_class_percent) for class_size, center_class_percent in zip(CLASS_SIZES, center_percents)) for center_percents in th.t(PERC)]
    total:          int       = sum(center_sizes)
    center_weights: th.Tensor = (th.tensor(center_sizes, dtype=th.float) / total)

    # move all to cpu since they could be in different GPUs
    net_weights: list[dict[str, th.Tensor]] = [{ a:b.cpu() for a,b in w.items() } for w in net_weights]

    printd("---------------------------------------")
    printd("Sizes for aggregated:", center_sizes)
    printd("---------------------------------------")

    # Perform a deep copy to retrieve just the structure, not the original values
    new_weights: dict[str, th.Tensor] = copy.deepcopy(net_weights[0])

    for key in new_weights.keys():
        stacked_tensors: th.Tensor = th.stack([state_dict[key] for state_dict in net_weights])
        new_weights[key] = (stacked_tensors * center_weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)

    printd("---------------------------------------")
    for i in range(N_CENTERS):
        printd("Saving in",  f"{RESULTS_PATH}/weights_{i}.pt")
        th.save(new_weights, f"{RESULTS_PATH}/weights_{i}.pt")
    printd("---------------------------------------")

if __name__ == '__main__':
    main()
