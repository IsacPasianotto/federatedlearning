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
import settings as S


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
            print("procs_join")
    # Ensure processes are terminated if something goes wrong
    except Exception as e:
        print(f"An error occurred: {e}")
        # Ensure processes are terminated
        for p in train_procs:
            p.terminate()
            p.join()

####
## Define main function
####

def main():
    mp.set_start_method('spawn')
    if th.cuda.is_available():
        nDevices = th.cuda.device_count()
        print(f"Found {nDevices} CUDA device(s)")
    else:
        raise RuntimeError("No CUDA devices found. This script requires GPU support. Aborting")

    # parameters:
    batch_size = S.BATCH_SIZE
    nEpochs = S.N_EPOCHS
    lr = S.LEARNING_RATE
    weight_decay = S.WEIGHT_DECAY
    agg_file = S.AGG_FILE

    if S.IMPORT_DATA:
        datasets = S.DATASETS
        if len(datasets) != nDevices:
            raise ValueError(f"Number of data files ({len(datasets)}) does not match the number of devices ({nDevices})")
        data = [Dataset(files=files) for files in datasets]
    else:
        allData = th.load('data/BrainCancerDataset.pt')
        datasets = allData.splitClasses(S.PERC)
        data = [Dataset(files=[dataset[i] for dataset in datasets]) for i in range(nDevices)]

    # Load datasets
    train_data, val_data, test_data = zip(*[dataset.train_val_test_split() for dataset in data])
    train_loaders = [build_Dataloader(d, batch_size) for d in train_data]
    val_loaders   = [build_Dataloader(d, batch_size) for d in val_data]
    test_loaders  = [build_Dataloader(d, batch_size) for d in test_data]

    return_dict = mp.Manager().dict()
    exec_procs( [mp.Process(target = train,
                            args = (i, networks.BrainClassifier(), train_loaders[i], val_loaders[i], return_dict, nEpochs, lr, weight_decay)
                            ) for i in range(nDevices)] )
    # Once all center has trained theyr own network, aggregate the results
    th.cuda.synchronize()  # be sure that everyone has finish his duty before starting the aggregation
    # Collect results from all processes
    results = [return_dict[i] for i in range(nDevices)]
    # Create the final models with the trained weights
    final_models = [networks.BrainClassifier() for _ in range(nDevices)]
    for model, result in zip(final_models, results):
        model.load_state_dict(result['state_dict'])

    # Evaluate all models on their respective test sets
    exec_procs( [mp.Process(target = test,
                            args = (i, final_models[i], test_loaders[i], f'Accuracy of the final model {i + 1} on its test set: ')
                            ) for i in range(nDevices)] )

    trainSizes = [len(train_data[i]) for i in range(nDevices)]
    evalAggregate(agg_file, results, trainSizes, batch_size)



if __name__ == '__main__':
    model = networks.BrainClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    main()

