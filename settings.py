import torch as th

# Training parameters
BATCH_SIZE=64
N_EPOCHS=5
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001

VERBOSE=True
DEBUG=False
PRINTWEIGHTS=False

NITER_FED = 5

DATA_PATH = './data'

RESULTS_PATH = './results'

# file that contains all data, to be distributed among data
#DOWNLOAD_URL = 'https://figshare.com/ndownloader/articles/1512427/versions/5'
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset?datasetVersionNumber=1"

#ALLDATA = DATA_PATH + 'BrainCancerDataset.pt'
ALLDATA = DATA_PATH + '/BrainCancerDatasetNew.pt'

#ZIP_FILE = DATA_PATH + 'BrainCancer.zip'
ZIP_FILE = DATA_PATH + '/BrainCancerNew.zip'

#EXTRACT_DIR = DATA_PATH + 'BrainCancerRawData/'
EXTRACT_DIR = DATA_PATH + '/BrainCancerRawDataNew/'

#DOTMAT_DIR = DATA_PATH + 'BrainCancerDotMat/'
DOTMAT_DIR = DATA_PATH + '/BrainCancerDotMatNew/'

#FILE_EXT='.mat'
FILE_EXT = '.jpg'
PIC_SQUARE_SIZE = 512   # the most common size

#perc must have shape (nCenters, nClasses): each tensor represents the percentages in which to split each class

PERC = th.tensor([[0.4, 0.3, 0.2, 0.5],     # Center 1
                 [0.3, 0.1, 0.4, 0.2],     # Center 2
                 [0.2, 0.4, 0.3, 0.1],     # Center 3
                 [0.1, 0.2, 0.1, 0.2],     # Center 4
                 ])

LABELS = {
        'notumor': 0,
        'meningioma': 1,
        'glioma': 2,
        'pituitary': 3
        }

NCENTERS = len(PERC)
NCLASSES = len(LABELS)

# The code was originally written for PERC written in the transposed form
PERC = th.t(PERC)

TRAINSIZE = 0.7
VALSIZE = 0.15
TESTSIZE = 0.15



CLASS_SIZES = [708, 1426, 915]

def printv(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)