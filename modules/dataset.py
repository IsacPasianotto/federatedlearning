########
## Imports
########

# Installed Modules
import os
import sys
import warnings
import torch as th
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Defined Modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from settings import *

########
## Dataset Class Definition
########

class Dataset(Dataset):
    """Dataset class for the used data. Extends the torch Dataset class.

    Class that represents the dataset used in the project. It is a subclass of the torch Dataset class, and it is used to store the images and labels of the dataset.
    It implements also the methods to split the dataset in train, validation and test set and to split the whole dataset in partition of it to emulate the federated learning scenario.
    Note that one between (images, labels) or files parameters should be provided, if both are provided, the files will be used.

    Parameters
    ----------
    images: th.Tensor, optional
        torch tensor of shape (nImages, nChannels, height, width) containing the images of the dataset
    labels: th.Tensor, optional
        torch tensor of shape (nImages) containing the labels of the dataset
    files: List[various], optional
        list of files to import data from (either strings or Dataset objects)

    Raises
    ------
    ValueError
        raised if neither images and labels nor files are provided
    ValueError
        raised if the number of percentages arrays is different from the number of classes
    ValueError
        raised if the sum of the percentages is not 1 for each class
    ValueError
        raised if the file does not exist, or if the file does not contain a Dataset object
    ValueError
        raised if the file does not exist
    ValueError
        raised if the file does not contain a Dataset object
    """

    def __init__(
        self,
        images: th.Tensor = None,
        labels: th.Tensor = None,
        files:  list      = None
    ):

        if images is None and labels is None and files is None:
            raise ValueError("Either images and labels or a list of files should be provided")

        if files is not None:
            self.images: th.Tensor = th.FloatTensor()
            self.labels: th.Tensor = th.LongTensor()
            self.importFromFiles(files)
        elif images is not None and labels is not None:
            self.images: th.Tensor = images.float()
            self.labels: th.Tensor = labels.long()

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        """Return the image and label at the given index

        Parameters
        ----------
        idx: int, required
            index of the image to be returned
        """
        return self.images[idx], self.labels[idx]

    def labelStr(
            self,
            label: int
    ) -> str:
        """Return the string representation of the given label

        Parameters
        ----------
        label: int, required
            label to be converted
        """
        for lab, n in LABELS.items():
            if n == label:
                return lab

    def shuffle(self) -> None:
        """Shuffle the dataset"""
        idx: th.Tensor = th.randperm(self.__len__())
        self.images: th.Tensor = self.images[idx]
        self.labels: th.Tensor = self.labels[idx]



    def train_val_test_split(
            self,
            train_percentage: float = TRAINSIZE,
            val_percentage:   float = VALSIZE,
            test_percentage:  float = TESTSIZE
    ) -> list:
        """Split the dataset in train, validation and test sets with the given percentages and return them
           as a list of this class objects

        Parameters
        ----------
        train_percentage: float, optional
            percentage of the dataset to be used as training set
        val_percentage: float, optional
            percentage of the dataset to be used as validation set
        test_percentage: float, optional
            percentage of the dataset to be used as test set

        Returns
        -------
        list
            list containing the train, validation and test sets
        """

        if abs(train_percentage + val_percentage + test_percentage -1) > 10E-6:
            train_percentage: float = 0.7
            val_percentage:   float = 0.15
            test_percentage:  float = 0.15
            print(f"Error with train-val-test percentages, using default values: {train_percentage} {val_percentage} {test_percentage}")

        self.shuffle()

        train_size: int = int(train_percentage * len(self))
        val_size:   int = int(val_percentage * len(self))
        test_size:  int = len(self) - train_size - val_size
        return random_split(self, [train_size, val_size, test_size])


    def splitClasses(
            self,
            percentPerClass: list[list[float]],
            save:            bool = False
    ) -> list:

        """Split the dataset into multiple datasets, one for each class and center, and saves them if desired

        Parameters
        ----------
        percentPerClass: (List[List[float]]), required
            List of lists of floats representing the percentage of each class to be in each subset
        save: bool, optional
            Decide if to save the created datasets or not. Default is False

        Raises
        ------
            ValueError: raised if the number of percentages arrays is different from the number of classes
            ValueError: raised if the sum of the percentages is not 1 for each class

        Returns
        -------
            List[List[Dataset]]: List of the created datasets, one for each class and center
        """
        datasets: list = [Subset(self, th.where(self.labels == i)[0]) for i in range(NCLASSES)]

        if len(percentPerClass) != NCLASSES:
            raise ValueError(f"The number of percentages for each center should be equal to the number of classes ({NCLASSES}), fill with zeroes if you don't want to consider some classes")
        if not th.all(th.isclose(th.sum(percentPerClass, dim=1), th.ones(1))):
            raise ValueError(f"The sum of the percentages of each class should be 1, but got {th.sum(percentPerClass, dim=1)}")

        output: np.ndarray = np.empty((NCLASSES, NCENTERS), dtype=object)

        for i in range(NCLASSES):
            # Ignore the warning about the 0 length of splits if there are percentages equal to 0
            #warnings.filterwarnings("ignore", category=UserWarning, message="Length of split.*")

            data: list = random_split(datasets[i], percentPerClass[i])
            # debugs, triggered only if options.DEBUG is True
            printd(percentPerClass[i])
            printd(len(datasets[i]), [len(d) for d in data])

            for j, subset in enumerate(data):

                # Extract data from the subset
                if len(subset) > 0:
                    images: th.Tensor = th.stack([subset.dataset[idx][0] for idx in subset.indices])
                    labels: th.Tensor = th.tensor([subset.dataset[idx][1] for idx in subset.indices])

                    # Create a new dataset
                    newData: Dataset = Dataset(images, labels)
                    output[i,j] = newData

                    if save:
                        label: str = self.labelStr(i)
                        os.makedirs(f"data/{label}", exist_ok=True)
                        th.save(newData, f"data/{label}/{label}{int(percentPerClass[i][j]*100)}_{j}.pt")

        return output.T

    def importFromFiles(
            self,
            files: list
    ) -> None:
        """ Import data from the given files (one for each class)

        Parameter
        ---------
        files: List[various], required
            List of files to import data from (either strings or Dataset objects)

        Raises
        ------
        ValueError
            raised if the file does not exist, or if the file does not contain a Dataset object
        """

        imgs: list = []
        labs: list = []

        for file in files:

            if file is not None:
                if isinstance(file, str):

                    if not os.path.isfile(file):
                        raise ValueError(f"File {file} does not exist")

                    data: Dataset = th.load(file)
                    if not isinstance(data, Dataset):
                        raise ValueError(f"File {file} does not contain a Dataset object")

                elif isinstance(file, Dataset):
                    data: Dataset = file

                else:
                    raise ValueError(f"Invalid file type: {type(file)}")

                imgs.append(data.images)
                labs.append(data.labels)

        self.images: th.Tensor = th.cat(imgs)
        self.labels: th.Tensor = th.cat(labs)
        self.shuffle()

        printd("nImgs:", len(self.images))  #debug




########
## Auxiliary data-related functions
########

def build_Dataloader(
        data:       th.Tensor,
        batch_size: int = BATCH_SIZE
    ) -> DataLoader:
    """ Build a DataLoader object for the given data

    Parameters
    ----------
    data: th.Tensor, required
        The data to be loaded
    batch_size: int, optional
        The batch size to be used. Default is taken from settings file

    Returns
    -------
    DataLoader: The DataLoader object
    """
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def buildDataloaders(data: Dataset) -> tuple:
    """ Split the given data in train, val, test sets and build their dataloaders

    Parameters
    ----------
    data: Dataset, required
        the data to be split
    """

    train_data, val_data, test_data = data.train_val_test_split()
    train_data: th.Tensor
    val_data:   th.Tensor
    test_data:  th.Tensor

    printd("trdata:",len(train_data), "valdata:", len(val_data), "testdata:", len(test_data))

    train_loader: DataLoader = build_Dataloader(train_data)
    val_loader:   DataLoader = build_Dataloader(val_data)
    test_loader:  DataLoader = build_Dataloader(test_data)

    printd("nbatches: train:", len(train_loader), "val:", len(val_loader))
    printd("dataloaders size: train:", len(train_loader.dataset), "val:", len(val_loader.dataset))

    return train_loader, val_loader, test_loader

