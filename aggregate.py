import torch as th
import copy
import sys

from settings import *

def aggregate_weights(state_dicts, trainSizes):
    """ Aggregate the weights of multiple models into a single model performing a weighted average of the weights.
    The weight is based on the size of each dataset

    Args:
        state_dicts (List[nn.Module]): List of dictionaries containing the weights of the models to aggregate
        trainSizes (List[int]): size of the datasets used to train each model

    Returns:
        nn.Module: A new model with the aggregated weights
    """
    total = sum(trainSizes)
    weights = (th.tensor(trainSizes, dtype=th.float) / total)

    new_state_dict = copy.deepcopy(state_dicts[0])

    for k in new_state_dict.keys():
        if 'num_batches_tracked' in k:
            # Special handling for BatchNorm layers
            new_state_dict[k] = state_dicts[0][k]
        else:
            stacked_tensors = th.stack([state_dict[k] for state_dict in state_dicts])
            new_state_dict[k] = (stacked_tensors * weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)

    return new_state_dict
    

def main():
    nGPUs = sys.argv[1]
    nNodes = sys.argv[2]
    weights = [[th.load(f"results/GPU{i}_Node{j}.pt") for i in range(nGPUs)] for j in range(nNodes)]
    nCenters = nGPUs * nNodes
    trainSizes = [sum(CLASS_SIZES[i] * PERC[i][j] * TRAINSIZE for i in range(len(CLASS_SIZES))) for j in range(nCenters)] 
    printd("Trainsizes for aggregated:", trainSizes)
    new_weights = aggregate_weights(weights, trainSizes)
    for i in range(nGPUs):
        for j in range(nNodes):
            print("Saving in", f"results/GPU{i}_Node{j}.pt")
            th.save(new_weights, f"results/GPU{i}_Node{j}.pt")
    
    
       
    
