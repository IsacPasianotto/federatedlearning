########
## Imports
########

import torch as th
from torch import Tensor

########
## Neural Network settings
########

BATCH_SIZE:    int   = 64
N_EPOCHS:      int   = 50
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY:  float = 1e-4
TRAINSIZE:     float = 0.7
VALSIZE:       float = 0.15
TESTSIZE:      float = 0.15

########
## Data settings
########

DATA_PATH:    str  = './data'
RESULTS_PATH: str  = './results/unbalanced_center-balanced_proportions'
AUGMENT_DATA: bool = True

# To generate the initial Dataset:
DOWNLOAD_URL:    str = "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=1"
ALLDATA:         str = DATA_PATH + '/BrainCancerDataset.pt'
ZIP_FILE:        str = DATA_PATH + '/BrainCancer.zip'
EXTRACT_DIR:     str = DATA_PATH + '/BrainCancerRawData/'
FILE_EXT:        str = '.jpg'
PIC_SQUARE_SIZE: int = 512

LABELS: dict[str, int] = {
    'notumor': 0,
    'meningioma': 1,
    'glioma': 2,
    'pituitary': 3
}

########
## Federated Learning settings
########

NITER_FED: int = 30

# PERC must have shape (nCenters, nClasses): each tensor represents the percentages in which to split each class

PERC: Tensor = th.tensor(
    [[0.25, 0.25, 0.25, 0.25],      # Center 1
     [0.25, 0.25, 0.25, 0.25],      # Center 2
     [0.1, 0.1, 0.1, 0.1],          # Center 3
     [0.4, 0.4, 0.4, 0.4],          # Center 4
    ])

NCENTERS: int = len(PERC)
NCLASSES: int = len(LABELS)

#Automatically updated by src/data_downloader.py
CLASS_SIZES: list[int] = [4000, 3290, 3242, 3514]

# The code was originally written for PERC written in the transposed form
PERC = th.t(PERC)

########
## Debug & Other settings
########

VERBOSE:      bool = True
DEBUG:        bool = False
PRINTWEIGHTS: bool = False

def printw(*args, **kwargs) -> None:
    """
    Print the weights of the model if PRINTWEIGHTS setting is True
    """
    if PRINTWEIGHTS:
        print(*args, **kwargs, flush=True)

def printv(*args, **kwargs) -> None:
    """
    Print the message if VERBOSE setting is True
    """
    if VERBOSE:
        print(*args, **kwargs, flush=True)

def printd(*args, **kwargs) -> None:
    """
    Print the message if DEBUG setting is True
    """
    if DEBUG:
        print(*args, **kwargs, flush=True)
