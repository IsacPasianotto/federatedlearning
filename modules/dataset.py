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

class BrainDataset(Dataset):
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
            self.import_from_files(files)
        elif images is not None and labels is not None:
            self.images: th.Tensor = images.float()
            self.labels: th.Tensor = labels.long()
        self.shuffle()

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

    def labelStr(self, label: int) -> str:
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
        idx:         th.Tensor = th.randperm(self.__len__())
        self.images: th.Tensor = self.images[idx]
        self.labels: th.Tensor = self.labels[idx]



    def train_val_split(
            self,
            train_percentage: float = TRAIN_SIZE,
            val_percentage:   float = VAL_SIZE,
    ) -> list:
        """Split the dataset in train and validation sets with the given percentages and return them
           as a list of this class objects

        Parameters
        ----------
        train_percentage: float, optional
            percentage of the dataset to be used as training set
        val_percentage: float, optional
            percentage of the dataset to be used as validation set

        Returns
        -------
        list
            list containing the train, validation and test sets
        """

        if abs(train_percentage + val_percentage + TEST_SIZE -1) > 10E-6:
            train_percentage: float = 0.7
            val_percentage:   float = 1 - train_percentage - TEST_SIZE
            print(f"Error with train-val percentages, using default values: {train_percentage} {val_percentage} {TEST_SIZE}")

        self.shuffle()
        train_total: int = train_percentage + val_percentage
        tr_perc:     int = train_percentage/train_total
        val_perc:    int = val_percentage/train_total
        warnings.filterwarnings("ignore", category=UserWarning, message="Length of split.*")
        return random_split(self, [tr_perc, val_perc])

    def get_class_sizes(self) -> th.Tensor:
        """Count the number of images for each class

        Returns
        -------
        th.Tensor
            tensor containing the number of images for each class
        """
        return th.bincount(self.labels, minlength=N_CLASSES)


    def split_classes(
            self,
            percent_per_class: th.FloatTensor,
            save:              bool = False
    ) -> np.ndarray:

        """Split the dataset into multiple datasets, one for each class and center, and saves them if desired

        Parameters
        ----------
        percent_per_class: (torch.FloatTensor), required
            List of lists of floats representing the percentage of each class to be in each subset
        save: bool, optional
            Decide if to save the created datasets or not. Default is False

        Raises
        ------
            ValueError: raised if the number of percentages arrays is different from the number of classes
            ValueError: raised if the sum of the percentages is not 1 for each class

        Returns
        -------
            np.ndarray: List of the created datasets, one for each class and center
        """
        datasets: list = [Subset(self, th.where(self.labels == i)[0]) for i in range(N_CLASSES)]

        if len(percent_per_class) != N_CLASSES:
            raise ValueError(f"The number of percentages for each center should be equal to the number of classes ({N_CLASSES}), fill with zeroes if you don't want to consider some classes")
        if not th.all(th.isclose(th.sum(percent_per_class, dim=1), th.ones(1))):
            raise ValueError(f"The sum of the percentages of each class should be 1, but got {th.sum(percent_per_class, dim=1)}")

        output: np.ndarray = np.empty((N_CLASSES, N_CENTERS), dtype=object)

        for i in range(N_CLASSES):
            # Ignore the warning about the 0 length of splits if there are percentages equal to 0
            warnings.filterwarnings("ignore", category=UserWarning, message="Length of split.*")

            data: list = random_split(datasets[i], percent_per_class[i])
            printd(percent_per_class[i])
            printd(len(datasets[i]), [len(d) for d in data])

            for j, subset in enumerate(data):

                # Extract data from the subset
                if len(subset) > 0:
                    images: th.Tensor = th.stack([subset.dataset[idx][0] for idx in subset.indices])
                    labels: th.Tensor = th.tensor([subset.dataset[idx][1] for idx in subset.indices])

                    # Create a new dataset
                    new_data: BrainDataset = BrainDataset(images, labels)
                    output[i,j] = new_data
                    if save:
                        label: str = self.labelStr(i)
                        os.makedirs(f"data/{label}", exist_ok=True)
                        th.save(new_data, f"data/{label}/{label}{int(percent_per_class[i][j]*100)}_{j}.pt")

        return output.T

    def import_from_files(
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

        printd("nImgs:", len(self.images))



########
## Auxiliary data-related functions
########

def build_Dataloader(
        data:       BrainDataset,
        batch_size: int = BATCH_SIZE
    ) -> DataLoader:
    """ Build a DataLoader object for the given data

    Parameters
    ----------
    data: BrainDataset, required
        The data to be loaded
    batch_size: int, optional
        The batch size to be used. Default is taken from settings file

    Returns
    -------
    DataLoader: The DataLoader object
    """
    if len(data) > 0:
        return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return None

def build_Dataloaders(
        data: BrainDataset,
        train_percentage: float = TRAIN_SIZE,
        val_percentage:   float = VAL_SIZE
    ) -> tuple:
    """ Split the given data in train, val, test sets and build their dataloaders

    Parameters
    ----------
    data: BrainDataset, required
        the data to be split
    train_percentage: float, optional
        the percentage of the data to be used as training set
    val_percentage: float, optional
        the percentage of the data to be used as validation set
    
    Returns
    -------
    tuple: The train and validation DataLoaders
    """
    train_data, val_data = data.train_val_split(train_percentage,val_percentage)
    train_data: th.Tensor
    val_data:   th.Tensor

    printd("trdata:",len(train_data), "valdata:", len(val_data))

    train_loader: DataLoader = build_Dataloader(train_data)
    val_loader:   DataLoader = build_Dataloader(val_data)

    printd("nbatches: train:", len(train_loader), "val:", len(val_loader))
    printd("dataloaders size: train:", len(train_loader.dataset), "val:", len(val_loader.dataset))

    return train_loader, val_loader
