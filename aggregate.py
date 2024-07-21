import torch as th
import copy
import sys

from settings import *
  
def main():
    printd("Computing aggregation")
    nGPUs = int(sys.argv[1])
    nNodes = int(sys.argv[2])
    net_weights = [th.load(f"results/GPU{i}_Node{j}.pt") for i in range(nGPUs) for j in range(nNodes)]
    net_weights = [{ a:b.cpu() for a,b in w.items() } for w in net_weights]
    nCenters = nGPUs * nNodes
    nClasses = len(CLASS_SIZES)
    trainSizes = [sum(int(CLASS_SIZES[i] * PERC[i][j] * TRAINSIZE) for i in range(nClasses)) for j in range(nCenters)] 
    total = sum(trainSizes)
    center_weights = (th.tensor(trainSizes, dtype=th.float) / total)
    printd("Trainsizes for aggregated:", trainSizes)
    new_weights = copy.deepcopy(net_weights[0])
    for k in new_weights.keys():
        stacked_tensors = th.stack([state_dict[k] for state_dict in net_weights])
        new_weights[k] = (stacked_tensors * center_weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)
    for i in range(nGPUs):
        for j in range(nNodes):
            printd("Saving in", f"results/GPU{i}_Node{j}.pt")
            th.save(new_weights, f"results/GPU{i}_Node{j}.pt")
    
    
if __name__ == '__main__':
    main()    
