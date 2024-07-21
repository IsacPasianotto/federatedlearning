####
## Imports
####

# Downloaded Modules
import torch as th
import torch.nn as nn
import torch.optim as optim

from settings import printv, printd
from modules.networks import BrainClassifier


####
## Defined functions
####

def train(model, device, centerID, train_data, val_data, nEpochs, lr=0.001, weight_decay=1e-4):
    """ Train the model on the given dataset

    Args:
        model (nn.Module): The model to be trained
        device (int): The rank of the GPU to be used
        centerID(int): The ID of the center
        train_data (th Dataloader): The training data
        val_data (th Dataloader): The validation data
        nEpochs (int): The number of epochs to train for
        lr (float, optional): The learning rate to use forprintv optimization. Defaults to 0.001.
        weight_decay (float, optional): The weight decay to use for optimization. Defaults to 1e-4.
    """
    th.cuda.set_device(device)
    printd("Training on", device, " for ",centerID, " with ", len(train_data), len(train_data.dataset), " and ", len(val_data), len(val_data.dataset))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create CUDA events for timing
    start = th.cuda.Event(enable_timing=True)
    end = th.cuda.Event(enable_timing=True)

    train_losses = []
    val_losses = []
    for epoch in range(nEpochs):
        model.train()
        start.record()
        epoch_loss = step(model, device, train_data, criterion, optimizer)
        end.record()

        train_losses.append(epoch_loss / len(train_data))

        model.eval()
        with th.no_grad():
            val_loss = step(model, device, val_data, criterion)
        val_losses.append(val_loss / len(val_data))
        printv(f"{device}, Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Time: {start.elapsed_time(end):.2f}ms")

    return model.state_dict()

def test(model, device, data, toPrint):
    """ Test the given model on the given data using the given device

    Args:
        rank (int): The rank of the GPU to be used
        model (nn.Module): the model to evaluate
        data (th Dataloader): the data to use to evaluate the model
        toPrint (str): the string to print before the accuracy
    """
    th.cuda.set_device(device)
    model.eval()
    correct = 0
    total = 0
    with th.no_grad():
        for image, label in data:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            correct += (outputs.argmax(1) == label).sum().item()
            total += label.size(0)
        accuracy = 100 * correct / total
    printv(toPrint + f'{accuracy:.2f}%')
    


def step(model, device, data, criterion, optimizer=None):
    """ Perform a single training or evaluation step

    Args:
        model (nn.Module): The model to be trained
        device (th.device): The device to be used
        data (DataLoader): The data to be used
        criterion (nn.Module): The loss function to be used
        optimizer (optim.Optimizer, optional): The optimizer to be used if training. Defaults to None (evaluation step).

    Returns:
        float: The total loss for the step
    """
    total_loss = 0
    for image, label in data:
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        outputs = model(image)
        loss = criterion(outputs, label)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss
