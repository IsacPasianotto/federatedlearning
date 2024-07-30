########
## Imports
########

# Installed modules:
import os
import sys
import torch as th

# Defined modules:
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from modules.dataset import BrainDataset
from settings import *


########
## Main
########

def main() -> None:

    printd("---------------------------------")
    printd("Removing previous center datasets")
    printd("---------------------------------")
    for f in os.listdir(DATA_PATH):
        if f.startswith("center_") and f.endswith(".pt"):
            os.remove(f"{DATA_PATH}/{f}")

    printd("-----------------------------------------------------")
    printd("Start generating independent datasets for each center")
    printd("-----------------------------------------------------")
    data:     th.Tensor     = th.load(ALL_DATA)
    tr_perc = TRAIN_SIZE + VAL_SIZE
    dataSplit = data.split_classes(th.tensor([[tr_perc]*N_CLASSES,
                                            [TEST_SIZE]*N_CLASSES,
                                            [0.0]*N_CLASSES,
                                            [0.0]*N_CLASSES]).T)                                       
    traindata = BrainDataset(files=dataSplit[0])
    testdata = BrainDataset(files=dataSplit[1])
    train_datasets: list[BrainDataset] = traindata.split_classes(PERC)
    train_new_data: list[BrainDataset] = [BrainDataset(files=d) for d in train_datasets]

    test_datasets: list[BrainDataset] = testdata.split_classes(PERC)
    test_new_data: list[BrainDataset] = [BrainDataset(files=d) for d in test_datasets]
    printd("-----------------------------------------------------")
    printd(f"Saving train: {len(train_new_data)} datasets with {[len(d) for d in train_new_data]} elements")
    printd("-----------------------------------------------------")
    for i,d in enumerate(train_new_data):
        th.save(d, f"{DATA_PATH}/center_{i}.pt")

    printd("-----------------------------------------------------")
    printd(f"Saving test:{len(test_new_data)} datasets with {[len(d) for d in test_new_data]} elements")
    printd("-----------------------------------------------------")
    for i,d in enumerate(test_new_data):
        th.save(d, f"{DATA_PATH}/center_{i}_test.pt")

if __name__ == '__main__':
    main()
    print("---------------------------------------------------------", flush = True)
    print("Finished to generate indipendent datasets for each center", flush = True)
    print("---------------------------------------------------------", flush = True)
