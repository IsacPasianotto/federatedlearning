####
##  Imports
####

# Downloaded Modules
import torch as th
import torch.multiprocessing as mp

# Defined Modules
from modules.dataset import buildData, buildDataloaders
import modules.networks as networks
from modules.traintest import train, test
from modules.federated import evalAggregate
from settings import *


####
## Defined functions
####

def exec_procs(train_procs):
    """ Execute all the given processes in parallel

    Args:
        train_procs (List[mp.Process]): the processes to execute
    """
    try:
        for p in train_procs:
            p.start()
        for p in train_procs:
            p.join()
    # Ensure processes are terminated if something goes wrong
    except Exception as e:
        print(f"An error occurred: {e}")
        for p in train_procs:
            p.terminate()
            p.join()
    #th.cuda.synchronize() # ensure that everyone has finished its work before starting the next batch

####
## Define main function
####

def main():
    ctx = mp.get_context("spawn")
    if th.cuda.is_available():
        nDevices = th.cuda.device_count()
        print(f"Found {nDevices} CUDA device(s) and {NCENTERS} center(s)")
    else:
        raise RuntimeError("No CUDA devices found. This script requires GPU support. Aborting")

    data = buildData()
    models = [networks.BrainClassifier() for _ in range(NCENTERS)]
    return_dict = mp.Manager().dict()
    for k in range(NITER_FED):
        print(f"################################ Starting iteration {k+1} ################################")
        results = {}
        # Generate the datasets
        train_loaders, val_loaders, test_loaders = buildDataloaders(data)
        for start in range(0, NCENTERS, nDevices):
            end = min(start + nDevices, NCENTERS)
            printd(f"doing from center {start+1} to {end} out of {NCENTERS}")
            exec_procs( [ctx.Process(target = train,
                                    args = (i%nDevices, i, models[i], train_loaders[i], val_loaders[i], return_dict, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
                                    ) for i in range(start,end)] )
            printd("return_dict has len",len(return_dict))
            results.update(return_dict)
            printd("update done, results has len", len(results))
   
        # Once all centers have trained their own network, aggregate the results
        # Create the final models with the trained weights
        for i in range(NCENTERS):
            models[i].load_state_dict(results[i]['state_dict'])

        # Evaluate all models on their respective test sets
        for start in range(0, NCENTERS, nDevices):
            end = min(start + nDevices, NCENTERS)
            exec_procs( [ctx.Process(target = test,
                                    args = (i%nDevices, models[i], test_loaders[i], 
                                            f'Accuracy of the final model {i + 1} on its test set: ')
                                    ) for i in range(start,end)] )
        trainSizes = [len(d.dataset) for d in train_loaders]
        aggrModel = evalAggregate(test_loaders, models, trainSizes)
        
        networks.printdParams(models + [aggrModel]) 
        
        aggr_dict = aggrModel.state_dict()
        for i in range(NCENTERS):
            models[i].load_state_dict(aggr_dict)
    


if __name__ == '__main__':
    model = networks.BrainClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    printd(f"Total number of parameters: {total_params}")
    main()
