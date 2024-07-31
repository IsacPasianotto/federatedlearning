# Downloaded Modules
import os
import sys
import torch as th

# Defined modules:
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from modules.dataset import BrainDataset, build_Dataloaders
from modules.networks import BrainClassifier
from modules.traintest import train_and_test
from settings import ALL_DATA, BASELINE_PATH

def main() -> None:
    print("Starting baseline")
    device: th.device = th.cuda.current_device()
    th.cuda.set_device(device)
    
    data: BrainDataset = th.load(ALL_DATA)
    print("Data loaded")
    model: BrainClassifier = BrainClassifier().to(device)
    train_loader, val_loader = build_Dataloaders(data, 0.8, 0.2)
    train_loader: th.utils.data.DataLoader
    val_loader:   th.utils.data.DataLoader
    
    print("Starting train")
    nEpochs = 300
    net_weights, train_losses, val_losses, accuracies = train_and_test(model, device, train_loader, val_loader, n_epochs=nEpochs, lr=1e-5, weight_decay=0)
    net_weights:  dict[str, th.Tensor]
    train_losses: th.Tensor
    val_losses:   th.Tensor
    accuracies:   th.Tensor
    
    print("Starting test")
    
    os.makedirs(BASELINE_PATH, exist_ok=True)
    save(train_losses, "train_losses.csv")
    save(val_losses, "val_losses.csv")
    save(accuracies, "accuracies.csv")


def save(values, filename):
    with open(BASELINE_PATH + filename, "w") as f:
        for val in values:
            f.write(str(float(val)))
            f.write("\n")

if __name__ == '__main__':
    main()
