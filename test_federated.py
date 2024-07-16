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

def aggregate_weights(state_dicts, lens):
    """
    Aggregate the weights of multiple models into a single model
    performing a weighted average of the weights
    The weighted is based on the size of each dataset
    :param state_dicts: state_dicts of the models
    :param lens: size of the datasets used to train each model
    :return:
    """
    total = sum(lens)
    weights = [len_ / total for len_ in lens]

    # Need to move to host all the weights because they are on different GPUs
    for i in range(len(state_dicts)):
        state_dicts[i] = {k: v.cpu() for k, v in state_dicts[i].items()}


    #aggregate the weights
    new_state_dict = {}
    for k in state_dicts[0].keys():
        new_state_dict[k] = sum([weights[i] * state_dicts[i][k] for i in range(len(state_dicts))])
    return new_state_dict



##########################################
##         Define the training          ##
##########################################
def train(rank, model, dataset, return_dict):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)

    train_loader, val_loader = dataset

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_losses = []
    val_losses = []

    # Create CUDA events for timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for epoch in range(5):  # 5 epochs just to test
        model.train()
        epoch_loss = 0
        start.record()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(
            f"GPU {rank}, Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Time: {start.elapsed_time(end):.2f}ms")

    return_dict[rank] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}  # Move to CPU
    }

def main():
    mp.set_start_method('spawn')

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s)")
        if device_count < 2:
            print("Warning: This script is designed to use 2 GPUs, but fewer were found.")
            return
    else:
        print("No CUDA devices found. This script requires GPU support.")
        return

    # Load datasets
    data1 = torch.load('data/BrainCancerDataset.pt')
    data2 = torch.load('data/BrainCancerDataset.pt')
    # ...
    data_agg = torch.load('data/BrainCancerDataset.pt')

    data1_train, data1_val, data1_test = data1.train_val_test_split()
    data2_train, data2_val, data2_test = data2.train_val_test_split()
    #...
    data_agg_train, data_agg_val, data_agg_test = data_agg.train_val_test_split()

    sizes = [len(data1_train), len(data2_train)]

    # create the data loaders
    train_loader1 = DataLoader(data1_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader1 = DataLoader(data1_val, batch_size=64, num_workers=4, pin_memory=True)
    test_loader1 = DataLoader(data1_test, batch_size=64, num_workers=4, pin_memory=True)

    train_loader2 = DataLoader(data2_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader2 = DataLoader(data2_val, batch_size=64, num_workers=4, pin_memory=True)
    test_loader2 = DataLoader(data2_test, batch_size=64, num_workers=4, pin_memory=True)

    # ....

    # train_loader_agg = DataLoader(data_agg_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader_agg = DataLoader(data_agg_val, batch_size=64, num_workers=4, pin_memory=True)
    test_loader_agg = DataLoader(data_agg_test, batch_size=64, num_workers=4, pin_memory=True)

    datasets = [(train_loader1, val_loader1), (train_loader2, val_loader2)]

    manager = mp.Manager()
    return_dict = manager.dict()

    model1 = BrainClassifier()
    model2 = BrainClassifier()

    p1 = mp.Process(target=train, args=(0, model1, datasets[0], return_dict))
    p2 = mp.Process(target=train, args=(1, model2, datasets[1], return_dict))

    try:
        p1.start()
        p2.start()

        p1.join()
        p2.join()

        # Collect results from all processe
        results = [return_dict[i] for i in range(2)]

        # Create two final models with the trained weights
        final_model1 = BrainClassifier()
        final_model1.load_state_dict(results[0]['state_dict'])

        final_model2 = BrainClassifier()
        final_model2.load_state_dict(results[1]['state_dict'])

        # Evaluate both models on their respective test sets
        for i, (final_model, test_loader) in enumerate([(final_model1, test_loader1), (final_model2, test_loader2)]):
            device = torch.device(f"cuda:{i}")
            final_model.to(device)
            final_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = final_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the final model {i + 1} on its test set: {accuracy:.2f}%')

        # Aggregate the weights of the two models to see the federated model
        federated_model = BrainClassifier()
        # bring all the weights to the host and aggregate them
        model1.load_state_dict({k: v.cpu() for k, v in model1.state_dict().items()})
        model2.load_state_dict({k: v.cpu() for k, v in model2.state_dict().items()})
        model1.to('cpu')
        model2.to('cpu')


        federated_weights = aggregate_weights([results[i]['state_dict'] for i in range(2)], sizes)
        federated_model.load_state_dict(federated_weights)

        # Evaluate the federated model on the test sets
        # use the first GPU for evaluation
        device = torch.device("cuda:0")
        print(f"Testing the federated model on GPU {device}")
        federated_model.to(device)
        federated_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader_agg:
                images, labels = images.to(device), labels.to(device)
                outputs = federated_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy of the aggregated model {i + 1} on its test set: {accuracy:.2f}%')


    # Ensure processes are terminated if something goes wrong
    except Exception as e:
        print(f"An error occurred: {e}")
        # Ensure processes are terminated
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()

if __name__ == '__main__':
    main()