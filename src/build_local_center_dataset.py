########
## Imports
########

# Installed modules:
import torch as th

# Defined modules:
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from settings import ALLDATA, PERC, printd
from dataset import Dataset


########
## Main
########

def main() -> None:
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
        th.save(d, f"data/center{i}.pt")

if __name__ == '__main__':
    main()
