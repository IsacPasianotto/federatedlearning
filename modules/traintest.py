####
## Imports
####

# Downloaded Modules
import torch as th
import torch.nn as nn
import torch.optim as optim



####
## Defined functions
####

def train(rank, model, train_data, val_data, return_dict, nEpochs, lr=0.001, weight_decay=1e-4):
    """ Train the model on the given dataset

    Args:
        rank (int): The rank of the GPU to be used
        model (nn.Module): The model to be trained
        train_data (th Dataloader): The training data
        val_data (th Dataloader): The validation data
        return_dict (Manager.dict): The dictionary to store the results
        nEpochs (int): The number of epochs to train for
        lr (float, optional): The learning rate to use for optimization. Defaults to 0.001.
        weight_decay (float, optional): The weight decay to use for optimization. Defaults to 1e-4.
    """
    device = th.device(f"cuda:{rank}")
    th.cuda.set_device(device)
    print("Training on", device)
    model = model.to(device)

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
        print(f"GPU {rank}, Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Time: {start.elapsed_time(end):.2f}ms")

    return_dict[rank] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}  # Move to CPU
    }

def test(i, model, data, toPrint):
    """ Test the given model on the given data using the given device

    Args:
        i (int): device number
        model (nn.Module): the model to evaluate
        data (th Dataloader): the data to use to evaluate the model
    """
    device = th.device(f"cuda:{i}")
    model.to(device).eval()
    correct = 0
    total = 0
    with th.no_grad():
        for image, label in data:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            correct += (outputs.argmax(1) == label).sum().item()
            total += label.size(0)
        accuracy = 100 * correct / total
    print(toPrint + f'{accuracy:.2f}%')
    # free up everything
    # th.cuda.synchronize()  # maybe not needed --> todo check better
    th.cuda.empty_cache()
    


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
