import numpy as np

# Training parameters
BATCH_SIZE=64
N_EPOCHS=1
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001

VERBOSE=True
DEBUG=True
PRINTWEIGHTS=False

NITER_FED = 3

DATA_PATH = './data'

RESULTS_PATH = './results'

# file that contains all data, to be distributed among data
ALLDATA = DATA_PATH + '/BrainCancerDataset.pt'

#perc must have shape (nClasses, nCenters): each tensor represents the percentages in which to split each class
PERC = np.array([[0.4, 0.3, 0.2, 0.1],  # Meningioma (708)
                 [0.3, 0.1, 0.4, 0.2],  # Glioma (1426)
                 [0.2, 0.4, 0.3, 0.1] ] )# Pituitary tumor (915)

TRAINSIZE = 0.7
VALSIZE = 0.15
TESTSIZE = 0.15


NCENTERS = len(PERC[0])

CLASS_SIZES = [708, 1426, 915]

def printv(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def printd(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
