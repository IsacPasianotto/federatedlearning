####
## Imports
####

import os
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from settings import *


####
## Defined class
####

class Dataset(Dataset):
    def __init__ (self, images=None, labels=None, files=None):
        if files is not None:
            self.images = th.FloatTensor()
            self.labels = th.LongTensor()
            self.importFromFiles(files)
        elif images is not None and labels is not None:
            self.images = images.float()
            self.labels = labels.long()
        else:
            raise ValueError("Either images and labels or a list of files should be provided")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def labelStr(self,label):
            return "Meningioma" if label == 0 else "Glioma" if label == 1 else "Pituitary tumor" if label == 2 else "unknown"

    def shuffle(self):
        idx = th.randperm(self.__len__())
        self.images = self.images[idx]
        self.labels = self.labels[idx]

    def train_val_test_split(self, train_percentage=TRAINSIZE, val_percentage=VALSIZE, test_percentage=TESTSIZE):
        if abs(train_percentage + val_percentage + test_percentage -1) > 10E-6:
            train_percentage = 0.7
            val_percentage = 0.15
            test_percentage = 0.15
            print(f"Error with train-val-test percentages, using default values: {train_percentage} {val_percentage} {test_percentage}")
        self.shuffle()
        train_size = int(train_percentage * len(self))
        val_size = int(val_percentage * len(self))
        test_size = len(self) - train_size - val_size
        return random_split(self, [train_size, val_size, test_size])

    def separateClasses(self):
        nClasses = th.max(self.labels)+1
        return [Subset(self, th.where(self.labels == i)[0]) for i in range(nClasses)]

    def splitClasses(self, percentPerClass, save=False):
        """Split the dataset into multiple datasets, one for each class and center, and saves them if desired

        Args:
            percentPerClass (List[List[float]]): List of lists of floats representing the percentage of each class to be in each subset
            save (bool, optional): Decide if to save the created datasets or not. Defaults to False.

        Raises:
            ValueError: raised if the number of percentages arrays is different from the number of classes
            ValueError: raised if the sum of the percentages is not 1 for each class

        Returns:
            List[List[Dataset]]: List of the created datasets, one for each class and center
        """
        datasets = self.separateClasses()
        nClasses = len(datasets)
        if len(percentPerClass) != nClasses:
            raise ValueError(f"The number of percentages arrays should be equal to the number of classes ({nClasses})")
        for percentList in percentPerClass:
            if not np.isclose(np.sum(percentList), 1.0, atol=1e-6):
                raise ValueError(f"The sum of the percentages of each class should be 1, but got {sum(percentList)} in {percentList}")
        output = np.empty((nClasses, len(percentPerClass[0])), dtype=object)
        for i in range(nClasses):
            dataset_size = len(datasets[i])
            split_sizes = [int(p * dataset_size) for p in percentPerClass[i][:-1]]
            split_sizes.append(dataset_size - sum(split_sizes))  # Add the remaining elements to the last subset
            data = random_split(datasets[i], split_sizes)
            for j, subset in enumerate(data):
                # Extract data from the subset
                images = th.stack([subset.dataset[idx][0] for idx in subset.indices])
                labels = th.tensor([subset.dataset[idx][1] for idx in subset.indices])
                # Create a new dataset
                newData = Dataset(images, labels)
                output[i,j] = newData
                if save:
                    label = self.labelStr(i)
                    os.makedirs(f"dataBrain/{label}", exist_ok=True)
                    # Save the new dataset
                    th.save(newData, f"dataBrain/{label}/{label}{int(percentPerClass[i][j]*100)}_{j}.pt")
        return output.T

    def importFromFiles(self, files):
        """ Import data from the given files (one for each class)

        Args:
            files (List[various]): List of files to import data from (either strings or Dataset objects)

        Raises:
            ValueError: raised if the file does not exist, or if the file does not contain a Dataset object
        """
        imgs = []
        labs = []
        for file in files:
            if isinstance(file, str):
                if not os.path.isfile(file):
                    raise ValueError(f"File {file} does not exist")
                data = th.load(file)
                if not isinstance(data, Dataset):
                    raise ValueError(f"File {file} does not contain a Dataset object")
            elif isinstance(file, Dataset):
                data = file
            else:
                raise ValueError(f"Invalid file type: {type(file)}")
            imgs.append(data.images)
            labs.append(data.labels)
        self.images = th.cat(imgs)
        self.labels = th.cat(labs)
        self.shuffle()



def build_Dataloader(data, batch_size): 
    """ Build a DataLoader object for the given data

    Args:
        data (th Subset): The data to be loaded
        batch_size (int): The batch size to be used

    Returns:
        DataLoader: The DataLoader object
    """
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def buildDataloaders(data):
    train_data, val_data, test_data = data.train_val_test_split()
    printd("trdata:",len(train_data), "valdata:", len(val_data), "testdata:", len(test_data))
    train_loader = build_Dataloader(train_data, BATCH_SIZE)
    val_loader   = build_Dataloader(val_data, BATCH_SIZE)
    test_loader  = build_Dataloader(test_data, BATCH_SIZE)
    printd("nbatches: train:", len(train_loader), "val:", len(val_loader))
    printd("dataloaders size: train:", len(train_loader.dataset), "val:", len(val_loader.dataset))
    return train_loader,val_loader,test_loader