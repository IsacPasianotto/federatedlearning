##  Imports
import os
import torch
import torchvision as tv

##  constants
ZIP_FILE='BrainCancer.zip'
DATA_DIR = 'data/Brain Cancer'

def main():
    # check if ./data dir is empty (expected only the .gitkeep file)
    if len(os.listdir('./data')) > 1:
        print('Data already extracted!\nskipping ....')
    else:
        os.system(f'unzip {ZIP_FILE} -d ./data')
        print('\n\nData extracted!')

    DATA_PATHS = []  # list of test data paths
    for dirname, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            # check if is a file or a folder
            if os.path.isdir(os.path.join(dirname, filename)):
                continue
            DATA_PATHS.append(os.path.join(dirname, filename))

    # read the images as a tensor with torchvision
    images = [tv.io.read_image(img) for img in DATA_PATHS]


    # Assign the labels to the images (files are named with the correct classification)

    # code:
    #  1 --> brain glioma
    #  2 --> meningioma
    #  3 --> tumor
    data_len = len(DATA_PATHS)
    labels = torch.zeros(data_len)

    for i, img in enumerate(DATA_PATHS):
        if 'glioma' in img:
            labels[i] = 1
        elif 'menin' in img:
            labels[i] = 2
        elif 'tumor' in img:
            labels[i] = 3

    # Associate the labels with the images keeping the same order and create a tensor dataset
    dataset = torch.utils.data.TensorDataset(torch.stack(images), labels)
    # save the dataset:
    torch.save(dataset, 'data/brain_cancer_dataset.pt')


if __name__ == '__main__':
    main()