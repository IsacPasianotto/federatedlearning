import torch as th

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

#perc must have shape (nCenters, nClasses): each tensor represents the percentages in which to split each class

PERC = th.tensor([[0.4, 0.3, 0.2],     # Center 1
                 [0.3, 0.1, 0.4],     # Center 2
                 [0.2, 0.4, 0.3],     # Center 3
                 [0.1, 0.2, 0.3]      # Center 4
                 ])

NCENTERS = len(PERC)

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

