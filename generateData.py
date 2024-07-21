import torch as th
from settings import PERC, printd
from modules.dataset import Dataset

def main():
    printd("Start generating data")
    data = th.load("data/BrainCancerDataset.pt")
    datasets = data.splitClasses(PERC)
    newData = [Dataset(files=d) for d in datasets]
    for i,d in enumerate(newData):
        th.save(d, f"data/center{i}.pt")

if __name__ == '__main__':
    main()