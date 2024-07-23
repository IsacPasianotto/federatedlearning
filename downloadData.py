import os
import torch as th
import h5py

from modules.dataset import Dataset
from settings import *


def exploreDir(dir, fileExt):
    return [ os.path.join(dir, file) for _,_, files in os.walk(dir)
            for file in files
            if any(file.endswith(fileExt))
            ]

def buildImgs_Labs():
    images= []
    labels = []
    matfile_names = exploreDir(DOTMAT_DIR, '.mat')
    for matfile in matfile_names:
        with h5py.File(matfile, 'r') as f:
            image = th.tensor(f['cjdata/image'][:])
            label = th.tensor(f['cjdata/label'][:]).squeeze()
            # The labels in the dataset are 1-indexed, so we need to subtract 1
            label -= 1
            if image.shape[0] != 256:
                images.append(image)
                labels.append(label)
    images_th = th.stack(images).unsqueeze(1)
    labels_th = th.stack(labels)
    return images_th,labels_th

def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(ZIP_FILE):
        print(f"Downloading data from {DOWNLOAD_URL}")
        os.system('wget -O %s %s' % (ZIP_FILE, DOWNLOAD_URL))
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting data from {ZIP_FILE}")
        os.system('unzip %s -d %s' % (ZIP_FILE, EXTRACT_DIR))
        exploreDir(EXTRACT_DIR, '.zip')
                        
    dataset = Dataset(*buildImgs_Labs())
    print(f"Saving data to {ALLDATA}")
    th.save(dataset, ALLDATA)

if __name__ == '__main__':
    main()
