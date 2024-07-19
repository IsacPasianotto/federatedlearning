####
## Imports
####

import torch.nn as nn
import torch.nn.functional as F
from settings import DEBUG


####
## Defined models
####

class BrainClassifier(nn.Module):
    def __init__(self):
        super(BrainClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc_input_size = 64 * 64 * 64  # Ensure this calculation matches the output size of the conv layers

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 3)  # 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    
    
def print_param_sum(model):
    total_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Sum of all parameters: {total_sum}")

def print_layer_sums(model):
    for name, param in model.named_parameters():
        param_sum = param.sum().item()
        print(f"{name}: Sum of values = {param_sum}")
        
def printdParams(models):
    if DEBUG:
        for i, model in enumerate(models):
            toprint = f"Model {i}:" if i < len(models) - 1 else "Aggregated Model:"
            print(toprint)
            print_param_sum(model)
            print_layer_sums(model)
            print("---")
