########
## Imports
########

# Installed modules:
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
import zipfile

# Defined modules:
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from settings import *
from dataset import Dataset

########
## Auxiliary functions
########

def exploreDir(
        dir:     str,
        fileExt: str
    ) -> list[str]:
    """Explore a directory and all its subdirectories and return a list of files with a given extension

    Parameters
    ----------
    dir : str
        The directory to explore
    fileExt : str
        The extension of the files to look for
    """

    return [ os.path.join(root, file) for root,_, files in os.walk(dir) for file in files if file.endswith(fileExt) ]

# TODO --> Rename with something like "build_img_and_labds".
#          In this way it seems that it builds only the labels

def buildImgs_Labs(
        augment: bool = AUGMENT_DATA
    ) -> tuple[th.Tensor[th.float32], th.Tensor[th.long]]:
    """Build the images and labels tensors from the dataset.

       If augment is True, it will also augment the images duplicating them and applying some transformations to the copies.

    Parameters
    ----------
    augment : bool, optional
        Whether to augment the images or not. The default is taken from the settings file.
    """

    images:      list[th.Tensor[th.float32]] = []
    labels:      list[th.Tensor[th.long]]    = []
    image_files: list[str]                   = exploreDir(EXTRACT_DIR, FILE_EXT)

    # Original pictures are only uniformed in size and transformed to tensors
    transform: transforms.Compose = transforms.Compose([
        transforms.Resize((PIC_SQUARE_SIZE, PIC_SQUARE_SIZE)),
        transforms.ToTensor()
    ])

    if augmented:
        transform_augmented: transforms.Compose = transforms.Compose([
            transforms.Resize((PIC_SQUARE_SIZE, PIC_SQUARE_SIZE)),    # Resize images to a fixed size
            transforms.RandomHorizontalFlip(),                        # Apply random horizontal flip
            transforms.RandomVerticalFlip(),                          # Apply random vertical flip
            # transforms.RandomEqualize(),                            # Apply random equalization -> substitute with ColorJitter
            transforms.RandomRotation(15),                            # Apply random rotations
            transforms.ToTensor(),                                    # Convert images to PyTorch tensors
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.3),  # Apply random color jitter (callable on tensors only)
        ])


    for image_file in image_files:
        image: Image = Image.open(image_file).convert('RGB')
        label: int = LABELS[image_file.split('/')[-2]]

        images.append(transform(image))
        labels.append(label)

        if augment:
            images.append(transform_augmented(image))
            labels.append(label)

    # Stack all the tensors in a single tensor
    images: th.Tensor[th.float32] = th.stack(images)
    labels: th.Tensor[th.long]    = th.tensor(labels)

    return images, labels

########
## Main function
########


def main() -> None:

    os.makedirs(DATA_PATH, exist_ok=True)

    if not os.path.exists(ZIP_FILE):
        print("-------------------------------------------------")
        print(f"Downloading data from {DOWNLOAD_URL}")
        print("-------------------------------------------------")
        os.system('wget -q -O %s %s' % (ZIP_FILE, DOWNLOAD_URL))

    if not os.path.exists(EXTRACT_DIR):

        print(f"Extracting data from {ZIP_FILE}")
        print("-------------------------------------------------")
        os.makedirs(EXTRACT_DIR)
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

    print("-------------------------------------------------")
    print(f"Resizing all the images to the same size: {PIC_SQUARE_SIZE}x{PIC_SQUARE_SIZE}")
    print("-------------------------------------------------")
    dataset = Dataset(*buildImgs_Labs())

    print("-------------------------------------------------")
    print(f"Saving data to {ALLDATA}")
    print("-------------------------------------------------")
    th.save(dataset, ALLDATA)


# Run the main function if the script is called directly
if __name__=='__main__':
    main()

