########
## Imports
########

# Installed modules:
import os
import sys
import torch as th

# Defined modules:
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from settings import ALLDATA, PERC, DATA_PATH, printd
from dataset import Dataset


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
    data:     th.Tensor     = th.load(ALLDATA)
    datasets: list[Dataset] = data.splitClasses(PERC)
    newData:  list[Dataset] = [Dataset(files=d) for d in datasets]

    printd("-----------------------------------------------------")
    printd(f"Saving {len(newData)} datasets with {[len(d) for d in newData]} elements")
    printd("-----------------------------------------------------")

    for i,d in enumerate(newData):
        th.save(d, f"{DATA_PATH}/center_{i}.pt")

if __name__ == '__main__':
    main()
    print("-------------------------------------------------------", flush = True)
    print("Finish to generate indipendent datasets for each center", flush = True)
    print("-------------------------------------------------------", flush = True)
