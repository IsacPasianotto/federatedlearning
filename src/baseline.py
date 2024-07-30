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
    train_loader, val_loader, test_loader = build_Dataloaders(data, 0.8, 0.2, 0.0)
    train_loader: th.utils.data.DataLoader
    val_loader:   th.utils.data.DataLoader
    test_loader:  th.utils.data.Data
    
    print("Starting train")
    nEpochs = 300
    net_weights, train_losses, val_losses, accuracies = train_and_test(model, device, train_loader, val_loader, n_epochs=nEpochs, lr=1e-5, weight_decay=0)
    net_weights:  dict[str, th.Tensor]
    train_losses: th.Tensor
    val_losses:   th.Tensor
    accuracies:   th.Tensor
    
    print("Starting test")
    
    #os.mkdirs(BASELINE_PATH, exist_ok=True)
    
    with open(BASELINE_PATH + "train_losses.csv", "w") as f:
        for loss in train_losses:
            f.write(str(float(loss)))
            f.write("\n")
    
    with open(BASELINE_PATH + "val_losses.csv", "w") as f:
        for loss in val_losses:
            f.write(str(float(loss)))
            f.write("\n")

    with open(BASELINE_PATH + "accuracies.csv", "w") as f:
        for acc in accuracies:
            f.write(str(float(acc)))
            f.write("\n")

if __name__ == '__main__':
    main()
