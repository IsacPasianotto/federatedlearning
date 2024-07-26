########
## Imports
########

# Downloaded Modules
import os
import sys
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

# Defined Modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from settings import *


########
## Defined functions
########

def train(
        model:        nn.Module,
        device:       th.device,
        train_data:   th.utils.data.DataLoader,
        val_data:     th.utils.data.DataLoader,
        nEpochs:      int   = N_EPOCHS,
        lr:           float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY
    ) -> th.Tensor:
    """ Train the model on the given dataset

    Parameters
    ----------
    model : nn.Module
        The model to be trained
    device : th.device
        The device to be used
    train_data : th.utils.data.DataLoader
        The training data
    val_data : th.utils.data.DataLoader
        The validation data
    nEpochs : int, optional
        The number of epochs to train for, by default N_EPOCHS
    lr : float, optional
        The learning rate to use for optimization, by default LEARNING_RATE
    weight_decay : float, optional
        The weight decay to use for optimization, by default WEIGHT_DECAY

    Returns
    -------
    th.Tensor
        The state dictionary of the model after training
    """

    cuda.set_device(device)

    criterion: nn.Module       = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create CUDA events for timing
    start = cuda.Event(enable_timing=True)
    end   = cuda.Event(enable_timing=True)

    train_losses: th.tensor = th.empty(nEpochs)
    val_losses:   th.tensor = th.empty(nEpochs)

    for epoch in range(nEpochs):

        model.train()

        start.record()
        epoch_loss: float = step(model, device, train_data, criterion, optimizer)
        end.record()

        train_losses[nEpochs] = epoch_loss / len(train_data)

        model.eval()

        with th.no_grad():
            val_loss: float = step(model, device, val_data, criterion)

        val_losses[nEpochs] = val_loss / len(val_data)
        printd(f"{device}, Epoch {epoch + 1}: Train Loss: {train_losses[nEpochs]:.4f}, Val Loss: {val_losses[nEpochs]:.4f}, Time: {start.elapsed_time(end):.2f}ms")
        
    return model.state_dict(), train_losses, val_losses


def test(
        model: nn.Module,
        device: th.device,
        data: th.utils.data.DataLoader
   ) -> float:
    """ Test the given model on the given data using the given device

    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    device : th.device
        The device to be used
    data : th.utils.data.DataLoader
        The data to use to evaluate the model

    Returns
    -------
    float
        The accuracy of the model on the given data
    """

    cuda.set_device(device)
    model.eval()

    correct: int = 0
    total:   int = 0

    with th.no_grad():
        for image, label in data:
            image, label = image.to(device), label.to(device)
            outputs: th.Tensor = model(image)
            correct += (outputs.argmax(1) == label).sum().item()
            total += label.size(0)

    return 100 * correct / total

def step(model, device, data, criterion, optimizer=None):
    """ Perform a single training or evaluation step

    Parameters
    ----------
    model : nn.Module
        The model to be trained
    device : th.device
        The device to be used
    data : DataLoader
        The data to be used
    criterion : nn.Module
        The loss function to be used
    optimizer : optim.Optimizer, optional
        The optimizer to be used if training, by default None (evaluation step)

    Returns
    -------
    float
        The total loss for the step
    """

    total_loss: float = 0

    for image, label in data:

        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)

        outputs: th.Tensor = model(image)
        loss:    th.Tensor = criterion(outputs, label)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss

