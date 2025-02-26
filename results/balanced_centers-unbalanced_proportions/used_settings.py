########
## Imports
########

from torch import tensor, Tensor, t

########
## Neural Network settings
########

BATCH_SIZE:    int   = 64
N_EPOCHS:      int   = 50
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY:  float = 1e-4
TRAIN_SIZE:    float = 0.7
VAL_SIZE:      float = 0.15
TEST_SIZE:     float = 0.15

########
## Data settings
########

DATA_PATH:     str  = './data'
RESULTS_PATH:  str  = './results/balanced_centers-unbalanced_proportions'
BASELINE_PATH: str  = './results/baseline/'
UGMENT_DATA:   bool = True

# To generate the initial Dataset:
DOWNLOAD_URL: str = "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=1"
ALL_DATA:     str = DATA_PATH + '/BrainCancerDataset.pt'
ZIP_FILE:     str = DATA_PATH + '/BrainCancer.zip'
EXTRACT_DIR:  str = DATA_PATH + '/BrainCancerRawData/'
FILE_EXT:     str = '.jpg'
PIC_SIZE:     int = 512

LABELS: dict[str, int] = {
    'notumor': 0,
    'meningioma': 1,
    'glioma': 2,
    'pituitary': 3
}

########
## Federated Learning settings
########

N_ITER_FED: int = 30

# PERC must have shape (nCenters, nClasses): each tensor represents the percentages in which to split each class

PERC: Tensor = tensor(
    [[0.25, 0.55, 0.10, 0.10],          # Center 1
     [0.25, 0.10, 0.55, 0.10],          # Center 2
     [0.25, 0.10, 0.10, 0.55],          # Center 3
     [0.25, 0.25, 0.25, 0.25],          # Center 4
    ])

N_CENTERS: int = len(PERC)
N_CLASSES: int = len(LABELS)

#Automatically updated by src/data_downloader.py
CLASS_SIZES: list[int] = [4000, 3290, 3242, 3514]

# The code was originally written for PERC written in the transposed form
PERC = t(PERC)

########
## Debug & Other settings
########

VERBOSE:       bool = True
DEBUG:         bool = False
PRINT_WEIGHTS: bool = False

def printw(*args, **kwargs) -> None:
    """
    Print the weights of the model if PRINTWEIGHTS setting is True
    """
    if PRINT_WEIGHTS:
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
