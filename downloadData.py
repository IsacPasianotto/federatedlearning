####
## Imports
####
import os
from PIL import Image  # Import PIL
import torchvision.transforms as transforms  # Import torchvision transforms for image processing
import zipfile

from modules.dataset import Dataset
from settings import *

def exploreDir(dir, fileExt):
    return [ os.path.join(root, file) for root,_, files in os.walk(dir) for file in files if file.endswith(fileExt) ]

def buildImgs_Labs():
    images = []
    labels = []
    image_files = exploreDir(EXTRACT_DIR, FILE_EXT)
    
    transform = transforms.Compose([
        transforms.Resize((PIC_SQUARE_SIZE, PIC_SQUARE_SIZE)),  # Resize images to a fixed size
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

    transform_augmented = transforms.Compose([
        transforms.Resize((PIC_SQUARE_SIZE, PIC_SQUARE_SIZE)),  # Resize images to a fixed size
        transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
        transforms.RandomVerticalFlip(),  # Apply random vertical flip
        # transforms.RandomEqualize(),
        transforms.RandomRotation(15),  # Apply random rotations
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.3),
    ])
    
        
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')  # Open the image and convert to RGB
        images.append(transform(image)) # append the standard image
        images.append(transform_augmented(image)) # append the augmented image
        label = LABELS[image_file.split('/')[-2]] # image_file is a path like .../Testing/glioma/Te-gl_0010.jpg, 
        # with split we isolate the folder name, and then we select the correct dir name (glioma)
        labels.append(label)
        labels.append(label)  # 2 labels for 2 images, standard and transformed
    images = th.stack(images)
    labels = th.tensor(labels)
    print(images.shape)
    return images, labels

def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(ZIP_FILE):
        print(f"Downloading data from {DOWNLOAD_URL}")
        os.system('wget -O %s %s' % (ZIP_FILE, DOWNLOAD_URL))
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting data from {ZIP_FILE}")
        os.makedirs(EXTRACT_DIR)
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    print("Resizing images")
    dataset = Dataset(*buildImgs_Labs())
    print(f"Saving data to {ALLDATA}")
    th.save(dataset, ALLDATA)

if __name__=='__main__':
    main()
