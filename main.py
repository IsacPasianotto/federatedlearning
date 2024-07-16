####
##  Imports
####

# Downloaded Modules
import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.multiprocessing as mp
from dotenv import load_dotenv

# Defined Modules
from modules.dataset import Dataset
import modules.networks as networks

####
## Global constants
####
load_dotenv()

BATCH_SIZE: int = int(os.getenv("BATCH_SIZE"))
N_EPOCHS: int = int(os.getenv("N_EPOCHS"))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE"))
WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY"))
IMPORT_DATA: bool = True if str(os.getenv("IMPORT_DATA")).lower() == "true" else False
AGG_FILE: str = str(os.getenv("AGG_FILE"))


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
    batch_size = BATCH_SIZE
    nEpochs = N_EPOCHS
    lr = LEARNING_RATE
    importData = IMPORT_DATA
    weight_decay = WEIGHT_DECAY
    agg_file = AGG_FILE # which file should we use for the aggregation evaluation?

    if importData:
        basePath = 'dataBrain/'
        # datasets must have shape (nDevices, nClasses), parameters and the way filenames are passed can be changed later
        datasets = [ [basePath+'Meningioma/Meningioma60_0.pt', basePath+'Glioma/Glioma60_0.pt', basePath+'Pituitary tumor/Pituitary tumor60_0.pt'],
                     [basePath+'Meningioma/Meningioma30_1.pt', basePath+'Glioma/Glioma30_1.pt', basePath+'Pituitary tumor/Pituitary tumor30_1.pt'],
                     [basePath+'Meningioma/Meningioma10_2.pt', basePath+'Glioma/Glioma10_2.pt', basePath+'Pituitary tumor/Pituitary tumor10_2.pt'] ]
        if len(datasets) != nDevices:
            raise ValueError(f"Number of data files ({len(datasets)}) does not match the number of devices ({nDevices})")
        data = [Dataset(files=files) for files in datasets]
    else:
        allData = th.load('data/BrainCancerDataset.pt')
        #perc must have shape (nClasses, nDevices) and sum to 1 for each class
        perc = [th.tensor([0.4, 0.3, 0.2, 0.1]),  # Meningioma
                th.tensor([0.3, 0.1, 0.4, 0.2]),  # Glioma
                th.tensor([0.2, 0.4, 0.1, 0.3]) ] # Pituitary tumor
        datasets = allData.splitClasses(perc)
        data = [Dataset(files=[dataset[i] for dataset in datasets]) for i in range(nDevices)]

    # Load datasets
    train_data, val_data, test_data = zip(*[data[i].train_val_test_split() for i in range(nDevices)])
    train_loaders = [build_Dataloader(d, batch_size) for d in train_data]
    val_loaders   = [build_Dataloader(d, batch_size) for d in val_data]
    test_loaders  = [build_Dataloader(d, batch_size) for d in test_data]

    return_dict = mp.Manager().dict()
    exec_procs( [mp.Process(target = train,
                            args = (i, BrainClassifier(), train_loaders[i], val_loaders[i], return_dict, nEpochs, lr, weight_decay)
                            ) for i in range(nDevices)] )

    # Collect results from all processes
    results = [return_dict[i] for i in range(nDevices)]
    # Create the final models with the trained weights
    final_models = [BrainClassifier() for _ in range(nDevices)]
    for model, result in zip(final_models, results):
        model.load_state_dict(result['state_dict'])

    # Evaluate all models on their respective test sets
    exec_procs( [mp.Process(target = test,
                            args = (i, final_models[i], test_loaders[i], f'Accuracy of the final model {i + 1} on its test set: ')
                            ) for i in range(nDevices)] )

    trainSizes = [len(train_data[i]) for i in range(nDevices)]
    evalAggregate(agg_file, results, trainSizes, batch_size)



if __name__ == '__main__':
    model = BrainClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    main()

