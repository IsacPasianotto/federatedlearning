########
## Imports
########

import torch
import torch.nn as nn
import torch.nn.functional as F

########
## Defined models
########

class BrainClassifier(nn.Module):
    """Simple CNN for classifying brain MRI images into 4 classes: notumor, meningioma, glioma, pituitary.

    The model consists of 3 convolutional layers followed by 3 fully connected layers.
    The network is thought to be used in a federated learning setting, for this reason the model
    is not too deep, to avoid long training times.
    The input size is [PIC_SQUARE_SIZE x PIC_SQUARE_SIZE x 3], which is the size of the RGB images.
    """

    def __init__(self):
        super(BrainClassifier, self).__init__()

        self.conv1: nn.Conv2d    = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2: nn.Conv2d    = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3: nn.Conv2d    = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten: nn.Flatten = nn.Flatten()

        self.fc_input_size: int = 64 * 64 * 64  # Ensure this calculation matches the output size of the conv layers

        self.fc1: nn.Linear = nn.Linear(self.fc_input_size, 16)
        self.fc2: nn.Linear = nn.Linear(16, 8)
        self.fc3: nn.Linear = nn.Linear(8, 4)
        self.fc4: nn.Linear = nn.Linear(4, 4)  # 4 out classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor, required
            Input tensor of shape [batch_size, 3, PIC_SQUARE_SIZE, PIC_SQUARE_SIZE]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, 4]
        """

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.flatten(x)
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

########
## Other utilities
########

def print_param_sum(model: nn.Module) -> None:
    """
    Print the sum of all parameters of the model. This can useful to check if the model is loaded correctly or the training is actually changing the weights.
    """
    total_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Sum of all parameters: {total_sum}")

def print_layer_sums(model: nn.Module) -> None:
    """
    Print the sum of all parameters of the model. This can useful to check if the model is loaded correctly or the training is actually changing the weights.
    """
    for name, param in model.named_parameters():
        param_sum = param.sum().item()
        print(f"{name}: Sum of values = {param_sum}")

def printParams(model: nn.Module) -> None:
    """
    Print the sum of all parameters of the model and the sum of all parameters of each layer.
    Used for debugging purposes.
    """
    print(f"Model:")
    print_param_sum(model)
    print_layer_sums(model)
    print("---")
