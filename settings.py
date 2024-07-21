import numpy as np

# Training parameters
BATCH_SIZE=64
N_EPOCHS=2
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001

VERBOSE=True
DEBUG=False
PRINTWEIGHTS=False

NITER_FED = 3
IMPORT_DATA=False

DATA_PATH = './data'

RESULTS_PATH = './results'

# file that contains all data, to be distributed among data
ALLDATA = './data/BrainCancerDataset.pt'

basePath = 'dataBrain/'
# datasets must have shape (nCenters, nClasses): each list represents the files to be used for a single center
DATASETS = [ [basePath+'Meningioma/Meningioma60_0.pt', basePath+'Glioma/Glioma60_0.pt', basePath+'Pituitary tumor/Pituitary tumor60_0.pt'],
             [basePath+'Meningioma/Meningioma30_1.pt', basePath+'Glioma/Glioma30_1.pt', basePath+'Pituitary tumor/Pituitary tumor30_1.pt'],
             [basePath+'Meningioma/Meningioma10_2.pt', basePath+'Glioma/Glioma10_2.pt', basePath+'Pituitary tumor/Pituitary tumor10_2.pt'] ]

#perc must have shape (nClasses, nCenters): each tensor represents the percentages in which to split each class
PERC = np.array([[0.4, 0.3, 0.2, 0.1],  # Meningioma (708)
                 [0.3, 0.1, 0.4, 0.2],  # Glioma (1426)
                 [0.2, 0.4, 0.3, 0.1] ] )# Pituitary tumor (915)

TRAINSIZE = 0.7
VALSIZE = 0.15
TESTSIZE = 0.15


NCENTERS = len(DATASETS) if IMPORT_DATA else len(PERC[0])

CLASS_SIZES = [708, 1426, 915]

def printv(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
