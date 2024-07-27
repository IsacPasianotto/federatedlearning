import os
import sys
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
# Defined modules:
from settings import *
from modules.dataset import Dataset, buildDataloaders
from modules.networks import BrainClassifier
from modules.traintest import *

def main():
    print("Starting baseline")
    device = th.cuda.current_device()
    th.cuda.set_device(device)
    
    data = th.load("data/BrainCancerDataset.pt")
    print("Data loaded")
    model = BrainClassifier().to(device)
    train_loader, val_loader, test_loader = buildDataloaders(data)
    print("Starting train")
    nEpochs = 150
    net_weights, train_losses, val_losses = train(model, device, train_loader, val_loader, nEpochs=nEpochs) 
    print("Starting test")
    acc = test(model, device, test_loader)
    print("Accuracy:", acc)
    
    resultsPath = "results/baselineResults/"
    os.mkdirs(resultsPath, exist_ok=True)
    
    with open(resultsPath + "train_losses.csv", "w") as f:
        for loss in train_losses:
            f.write(loss)
    
    with open(resultsPath + "val_losses.csv", "w") as f:
        for loss in val_losses:
            f.write(loss)

if __name__ == '__main__':
    main()