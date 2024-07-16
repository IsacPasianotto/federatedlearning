import torch as th

BATCH_SIZE=64
N_EPOCHS=5
LEARNING_RATE=0.001
IMPORT_DATA=False
WEIGHT_DECAY=0.0001

# which file should we use for the aggregation evaluation?
AGG_FILE='./data/BrainCancerDataset.pt'

basePath = 'dataBrain/'
# datasets must have shape (nDevices, nClasses)
DATASETS = [ [basePath+'Meningioma/Meningioma60_0.pt', basePath+'Glioma/Glioma60_0.pt', basePath+'Pituitary tumor/Pituitary tumor60_0.pt'],
             [basePath+'Meningioma/Meningioma30_1.pt', basePath+'Glioma/Glioma30_1.pt', basePath+'Pituitary tumor/Pituitary tumor30_1.pt'],
             [basePath+'Meningioma/Meningioma10_2.pt', basePath+'Glioma/Glioma10_2.pt', basePath+'Pituitary tumor/Pituitary tumor10_2.pt'] ]

#perc must have shape (nClasses, nDevices) and sum to 1 for each class
PERC = [th.tensor([0.4, 0.3, 0.2, 0.1]),  # Meningioma
        th.tensor([0.3, 0.1, 0.4, 0.2]),  # Glioma
        th.tensor([0.2, 0.4, 0.1, 0.3]) ] # Pituitary tumor