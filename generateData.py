import torch as th
from settings import ALLDATA, PERC, printd
from modules.dataset import Dataset

def main():
    printd("Start generating data")
    data = th.load(ALLDATA)
    datasets = data.splitClasses(PERC)
    newData = [Dataset(files=d) for d in datasets]
    printd(f"Saving {len(newData)} datasets with {[len(d) for d in newData]} elements") 
    for i,d in enumerate(newData):
        th.save(d, f"data/center{i}.pt")

if __name__ == '__main__':
    main()