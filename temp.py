import torch 
data = torch.load('data/brain_cancer_dataset.pt')
glioma_start, glioma_end = 0, 100
meningioma_start, meningioma_end = 5000, 5100
tumor_start, tumor_end = 10000, 10100

data_small = data[list(range(glioma_start, glioma_end)) + list(range(meningioma_start, meningioma_end)) + list(range(tumor_start, tumor_end))]

data_small_ds = torch.utils.data.TensorDataset(data_small[0], data_small[1])

torch.save(data_small_ds, 'data/brain_cancer_dataset_small.pt')

