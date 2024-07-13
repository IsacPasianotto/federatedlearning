import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp


##########################################
##       Define the dataset class       ##
##########################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.images = data['images'].float()
        self.labels = data['labels'].long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def shuffle(self):
        idx = torch.randperm(self.__len__())
        self.images = self.images[idx]
        self.labels = self.labels[idx]

    def train_val_test_split(self, train_percentage=0.7, val_percentage=0.15, test_percentage=0.15):
        self.shuffle()
        train_size = int(train_percentage * len(self))
        val_size = int(val_percentage * len(self))
        test_size = len(self) - train_size - val_size
        return torch.utils.data.random_split(self, [train_size, val_size, test_size])


##########################################
##          Define the model            ##
##########################################

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


###############################
##   Train-test functions    ##
###############################

def train(rank, model, train_data, val_data, return_dict, nEpochs, lr=0.001, weight_decay=1e-4):
    """ Train the model on the given dataset

    Args:
        rank (int): The rank of the GPU to be used
        model (nn.Module): The model to be trained
        train_data (torch Dataloader): The training data
        val_data (torch Dataloader): The validation data
        return_dict (Manager.dict): The dictionary to store the results
        nEpochs (int): The number of epochs to train for
        lr (float, optional): The learning rate to use for optimization. Defaults to 0.001.
        weight_decay (float, optional): The weight decay to use for optimization. Defaults to 1e-4.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print("Training on", device)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create CUDA events for timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    train_losses = []
    val_losses = []
    for epoch in range(nEpochs):
        model.train()
        start.record()
        epoch_loss = step(model, device, train_data, criterion, optimizer)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        train_losses.append(epoch_loss / len(train_data))

        model.eval()
        with torch.no_grad():
            val_loss = step(model, device, val_data, criterion)
        # torch.cuda.synchronize() # needed?
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
        data (torch Dataloader): the data to use to evaluate the model
    """
    device = torch.device(f"cuda:{i}")
    model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in data:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            correct += (outputs.argmax(1) == label).sum().item()
            total += label.size(0)
        accuracy = 100 * correct / total
    print(toPrint + f'{accuracy:.2f}%')


def step(model, device, data, criterion, optimizer=None):
    """ Perform a single training or evaluation step

    Args:
        model (nn.Module): The model to be trained
        device (torch.device): The device to be used
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

###############################
##      Other functions      ##
###############################

def build_Dataloader(data, batch_size): 
    """ Build a DataLoader object for the given data

    Args:
        data (torch Subset): The data to be loaded
        batch_size (int): The batch size to be used

    Returns:
        DataLoader: The DataLoader object
    """
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
 
 
def aggregate_weights(state_dicts, trainSizes):
    """ Aggregate the weights of multiple models into a single model performing a weighted average of the weights.
    The weighted is based on the size of each dataset
    
    Args:
        state_dicts (List[dict]): state_dicts of the models
        trainSizes (List[int]): size of the datasets used to train each model
    
    Returns:
        Dict: the dictionary with the weighted sum of the weights
    """
    total = sum(trainSizes)
    weights = (torch.tensor(trainSizes, dtype=torch.float) / total) # weights for each dataset, based on the size of the dataset, reshaped to be broadcastable
    #aggregate the weights
    new_state_dict = {}
    for k in state_dicts[0].keys():  # keys are the names the network nodes weights, biases, etc --> they are the same for all the networks since they are all identical
        stacked_tensors = torch.stack([state_dict[k] for state_dict in state_dicts])
        new_state_dict[k] = (stacked_tensors * weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)
        # explanation of the line above:
        # [1] * (stacked_tensors.dim() - 1) creates a list of 1s with the same length as the dimensions of the stacked_tensors minus 1
        # -1 is used to keep the dimensions of the weights tensor the same as the stacked_tensors tensor, with the first dimension inferred from the weights tensor
        # the weights tensor is reshaped to be broadcastable to the stacked_tensors tensor
        # note that state_dict[k] could be a bias or a weight, which have different dimensions
    return new_state_dict
   
    
def evalAggregate(data, results, trainSizes, batch_size):
    """ Evaluate the aggregated model on the test set

    Args:
        data (String): the file where to load data from
        results (List[Dict]): The results of the training process for each device
        trainSizes (List[int]): The sizes of the training datasets for each device
        batch_size (int): The batch size to be used
    """
    # Aggregate the weights of the two models to see the federated model
    federated_model = BrainClassifier()
    federated_weights = aggregate_weights([results[i]['state_dict'] for i in range(len(results))], trainSizes)
    federated_model.load_state_dict(federated_weights)

    data_agg = torch.load(data)
    data_agg_train, data_agg_val, data_agg_test = data_agg.train_val_test_split()
    # train_loader_agg = build_Dataloader(data_agg_train, batch_size)
    # val_loader_agg   = build_Dataloader(data_agg_val, batch_size)
    test_loader_agg    = build_Dataloader(data_agg_test, batch_size)
    test(0, federated_model, test_loader_agg, f'Accuracy of {len(results)} aggregated models on test set: ')
        
def exec_procs(train_procs):
    """ Execute all the given processes in parallel

    Args:
        train_procs (List[mp.Process]): the processes to execute
    """
    try:
        for p in train_procs:
            p.start()
        for p in train_procs:
            p.join()
    # Ensure processes are terminated if something goes wrong
    except Exception as e:
        print(f"An error occurred: {e}")
        # Ensure processes are terminated
        for p in train_procs:
            p.terminate()
            p.join()
    
#############################################################################    
#############################################################################
#############################################################################
#############################################################################

def main():
    mp.set_start_method('spawn')
    if torch.cuda.is_available():
        nDevices = torch.cuda.device_count()
        print(f"Found {nDevices} CUDA device(s)")
    else:
        print("No CUDA devices found. This script requires GPU support. Aborting")
        return
    
    # parameters:
    batch_size = 64    
    nEpochs = 5
    lr = 0.001
    weight_decay = 1e-4
    dataFiles = ['BrainCancerDataset.pt'] * nDevices # To be changed to have different data for each device!
    agg_file = 'data/BrainCancerDataset.pt'
    
    # Load datasets
    data = [torch.load('data/' + filename) for filename in dataFiles]
    train_data, val_data, test_data = zip(*[data[i].train_val_test_split() for i in range(nDevices)])
    train_loaders = [build_Dataloader(d, batch_size) for d in train_data]
    val_loaders   = [build_Dataloader(d, batch_size) for d in val_data]
    test_loaders  = [build_Dataloader(d, batch_size) for d in test_data]

    return_dict = mp.Manager().dict()
    exec_procs( [mp.Process(target = train, 
                            args = (i, BrainClassifier(), train_loaders[i], val_loaders[i], return_dict, nEpochs, lr, weight_decay)
                            ) for i in range(nDevices)] )
    
    # Collect results from all processe
    results = [return_dict[i] for i in range(nDevices)]
    # Create the final models with the trained weights
    final_models = [BrainClassifier() for _ in range(nDevices)]
    for model, result in zip(final_models, results):
        model.load_state_dict(result['state_dict'])

    # Evaluate all models on their respective test sets
    exec_procs( [mp.Process(target = test,  
                            args = (i, final_models[i], test_loaders[i], f'Accuracy of the final model {i + 1} on its test set: ')
                            ) for i in range(nDevices)] )
    
    trainSizes = [len(train_data[i]) for i in range(nDevices)]
    evalAggregate(agg_file, results, trainSizes, batch_size)
    
    

if __name__ == '__main__':
    main()
