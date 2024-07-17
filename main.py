####
##  Imports
####

# Downloaded Modules
import torch as th
import torch.multiprocessing as mp

# Defined Modules
from modules.dataset import Dataset, build_Dataloader
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

    # parameters:
    batch_size = BATCH_SIZE
    nEpochs = N_EPOCHS
    lr = LEARNING_RATE
    weight_decay = WEIGHT_DECAY
    agg_file = AGG_FILE
    nCenters = NCENTERS

    if IMPORT_DATA:
        datasets = [[th.load(file) for file in center] for center in DATASETS]
    else:
        if len(set(map(len, PERC)))>1:
            raise ValueError(f"Number of percentages should be the same for all centers")
        allData = th.load(ALLDATA)
        datasets = allData.splitClasses(PERC)
    printd("datasets:",[[len(d) for d in dat] for dat in datasets])
    data = [Dataset(files=d) for d in datasets]
    printd("data:",[len(d) for d in data])
    # Load datasets
    train_data, val_data, test_data = zip(*[dataset.train_val_test_split() for dataset in data])
    printd("trdata:",[len(t) for t in train_data], "valdata:", [len(v) for v in val_data], "testdata:", [len(v) for v in test_data])
    train_loaders = [build_Dataloader(d, batch_size) for d in train_data]
    val_loaders   = [build_Dataloader(d, batch_size) for d in val_data]
    test_loaders  = [build_Dataloader(d, batch_size) for d in test_data]

    return_dict = mp.Manager().dict()
    results = {}
    printd("nbatches: train:", [len(l) for l in train_loaders], "val:",[len(v) for v in val_loaders])
    printd("dataloaders size: train:", [len(l.dataset) for l in train_loaders], "val:",[len(v.dataset) for v in val_loaders])
    for start in range(0, nCenters, nDevices):
        end = min(start + nDevices, nCenters)
        printd(f"doing from center {start+1} to {end} out of {nCenters}")
        exec_procs( [ctx.Process(target = train,
                                args = (i%nDevices, i, networks.BrainClassifier(), train_loaders[i], val_loaders[i], return_dict, nEpochs, lr, weight_decay)
                                ) for i in range(start,end)] )
        printd("len return_dict: ",len(return_dict))
        results.update(return_dict)
        printd("done, res has", len(results))
        
    # Once all centers have trained their own network, aggregate the results
    # Create the final models with the trained weights
    final_models = [networks.BrainClassifier() for _ in range(nCenters)]
    for i in range(nCenters):
        final_models[i].load_state_dict(results[i]['state_dict'])

    # Evaluate all models on their respective test sets
    for start in range(0, nCenters, nDevices):
        end = min(start + nDevices, nCenters)
        exec_procs( [ctx.Process(target = test,
                            args = (i%nDevices, final_models[i], test_loaders[i], f'Accuracy of the final model {i + 1} on its test set: ')
                            ) for i in range(start,end)] )
    trainSizes = [len(d) for d in train_data]
    evalAggregate(agg_file, results, trainSizes, batch_size)
    


if __name__ == '__main__':
    # model = networks.BrainClassifier()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")
    main()
