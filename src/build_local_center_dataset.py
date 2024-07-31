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
    data: BrainDataset = th.load(ALL_DATA)
    dataSplit = data.split_classes(th.tensor([[TRAIN_SIZE+VAL_SIZE]*N_CLASSES,
                                                        [TEST_SIZE]*N_CLASSES,
                                                              [0.0]*N_CLASSES,
                                                              [0.0]*N_CLASSES]).T)                                       
    generate_and_save_data(dataSplit[0], False)
    generate_and_save_data(dataSplit[1], True)

    print("---------------------------------------------------------")
    print("Finished to generate indipendent datasets for each center")
    print("---------------------------------------------------------")



def generate_and_save_data(
    data: BrainDataset,
    isTest: bool
    ) -> None:
    new_data = BrainDataset(files=data)
    datasets: list[BrainDataset] = new_data.split_classes(PERC)
    new_data: list[BrainDataset] = [BrainDataset(files=d) for d in datasets]
    printd("-----------------------------------------------------")
    printd(f"Saving: {len(new_data)} datasets with {[len(d) for d in new_data]} elements")
    printd("-----------------------------------------------------")
    end = "_test" if isTest else ""
    for i,d in enumerate(new_data):
        th.save(d, f"{DATA_PATH}/center_{i}{end}.pt")


if __name__ == '__main__':
    main()
