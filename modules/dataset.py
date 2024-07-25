# Downloaded Modules
import os
import warnings
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Defined Modules
from settings import *


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
            for lab, n in LABELS.items():
                if n == label:
                    return lab

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

    def dbg(self, datasets, percentPerClass):
        lens = [len(d) for d in datasets]
        def dot(a,b):
            return sum(int(x*y) for x,y in zip(a,b))
        printd(f"Expected sizes: {[dot(lens, list(p)) for p in percentPerClass.T]}")
        
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
        datasets = [Subset(self, th.where(self.labels == i)[0]) for i in range(NCLASSES)]
        self.dbg(datasets, percentPerClass)
        if len(percentPerClass) != NCLASSES:
            raise ValueError(f"The number of percentages for each center should be equal to the number of classes ({NCLASSES}), fill with zeroes if you don't want to consider some classes")
        if not th.all(th.isclose(th.sum(percentPerClass, dim=1), th.ones(1))):
            raise ValueError(f"The sum of the percentages of each class should be 1, but got {th.sum(percentPerClass, dim=1)}")
        output = np.empty((NCLASSES, NCENTERS), dtype=object)
        for i in range(NCLASSES):
            #warnings.filterwarnings("ignore", category=UserWarning, message="Length of split.*") # Ignore the warning about the 0 length of splits if there are percentages equal to 0
            data = random_split(datasets[i], percentPerClass[i])
            printd(percentPerClass[i])
            printd(len(datasets[i]), [len(d) for d in data])
            for j, subset in enumerate(data):
                # Extract data from the subset
                if len(subset) > 0:
                    images = th.stack([subset.dataset[idx][0] for idx in subset.indices])
                    labels = th.tensor([subset.dataset[idx][1] for idx in subset.indices])
                    # Create a new dataset
                    newData = Dataset(images, labels)
                    output[i,j] = newData
                    if save:
                        label = self.labelStr(i)
                        os.makedirs(f"data/{label}", exist_ok=True)
                        # Save the new dataset
                        th.save(newData, f"data/{label}/{label}{int(percentPerClass[i][j]*100)}_{j}.pt")
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
            if file is not None:
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
        printd("nImgs:", len(self.images))
        self.shuffle()



def build_Dataloader(data, batch_size=BATCH_SIZE):
    """ Build a DataLoader object for the given data

    Args:
        data (th Subset): The data to be loaded
        batch_size (int): The batch size to be used

    Returns:
        DataLoader: The DataLoader object
    """
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def buildDataloaders(data):
    """ Split the given data in train, val, test sets and build their dataloaders

    Args:
        data (th.Dataset): the data to be split

    Returns:
        th.Dataloader, th.Dataloader, th.Dataloader: train, val, test dataloaders
    """
    train_data, val_data, test_data = data.train_val_test_split()
    printd("trdata:",len(train_data), "valdata:", len(val_data), "testdata:", len(test_data))
    train_loader = build_Dataloader(train_data)
    val_loader   = build_Dataloader(val_data)
    test_loader  = build_Dataloader(test_data)
    printd("nbatches: train:", len(train_loader), "val:", len(val_loader))
    printd("dataloaders size: train:", len(train_loader.dataset), "val:", len(val_loader.dataset))
    return train_loader, val_loader, test_loader

