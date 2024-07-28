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
from settings import ALL_DATA, PERC, DATA_PATH, printd


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
    datasets: list[BrainDataset] = data.split_classes(PERC)
    new_data: list[BrainDataset] = [BrainDataset(files=d) for d in datasets]

    printd("-----------------------------------------------------")
    printd(f"Saving {len(new_data)} datasets with {[len(d) for d in new_data]} elements")
    printd("-----------------------------------------------------")

    for i,d in enumerate(new_data):
        th.save(d, f"{DATA_PATH}/center_{i}.pt")

if __name__ == '__main__':
    main()
    print("-------------------------------------------------------", flush = True)
    print("Finish to generate indipendent datasets for each center", flush = True)
    print("-------------------------------------------------------", flush = True)
