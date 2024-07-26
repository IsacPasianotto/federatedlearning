# Federated learning for brain cancer classification

<p style="text-align: center;">

[![License](https://img.shields.io/badge/License-MIT%20license-mitlicense?style=for-the-badge&logo=MIT&color=%23FF9E0F)](https://github.com/IsacPasianotto/federatedlearning/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-Python?style=for-the-badge&logo=Python&logoColor=%23FECC00&color=%233776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-PyTorch?style=for-the-badge&logo=PyTorch&logoColor=white&color=%23EE4C2C)](https://pytorch.org/)

</p>

_Authors_:\
&emsp;[**Pasianotto Isac**](https://github.com/IsacPasianotto/)\
&emsp;[**Rossi Davide**](https://github.com/DavideRossi1/)


# TODO:

- [ ] More meaningfull names to some function and classes (eg Dataset, ...)
- [ ] Consistency... al in CamelCase or all in snake_case. (better snake)
- [ ] In the `DataSet` class: in the `splitClasses` method there is `numpy`
- [ ] Fix all the variable names in snake_case in `src/aggregate_weights.py`
- [ ] Subsitute the print of the accuracy of the various network with a way to save it in order to plot it later
- [ ] Some Nice plots
- [x] ~~Table with settings explained in detail~~
- [x] ~~Add a decent README file~~
  - [ ] Add a theoretical description in section 1.1
- [ ] Add some references at this readme files
- [ ] Remove this todo list

# 0. Table of contents

- [Federated learning for brain cancer classification](#federated-learning-for-brain-cancer-classification)
- [TODO:](#todo)
- [0. Table of contents](#0-table-of-contents)
- [1. Description](#1-description)
  - [1.0 What is the idea behind federated](#10-what-is-the-idea-behind-federated)
  - [1.1 How this code works](#11-how-this-code-works)
- [2. Getting started](#2-getting-started)
  - [2.0 Prerequisites](#20-prerequisites)
  - [2.1 Generate the dataset:](#21-generate-the-dataset)
- [3. How to run](#3-how-to-run)
- [4. Settings doc:](#4-settings-doc)
  - [4.0 Neural Network settings](#40-neural-network-settings)
  - [4.1 Data settings](#41-data-settings)
  - [4.2 Federated settings](#42-federated-settings)
  - [4.3 Other settings](#43-other-settings)
- [5. References:](#5-references)


# 1. Description

## 1.0 What is the idea behind federated

## 1.1 How this code works

- Structure of this repository
- Loop in the sbatch script


TODO --> fill this section 

# 2. Getting started

## 2.0 Prerequisites

All the presented code was tested with `Python 3.10.12`, using the libraries listed in the [`environment.txt`](./environment.txt) file. 

To get started with the code, we recommend to create a virtual with [`venv`](https://docs.python.org/3/library/venv.html) as follows:

```bash
python3 -m venv federatedenv
source $(pwd)/federatedenv/bin/activate
pip install -r environment.txt
```

## 2.1 Generate the dataset:

Before running the `main.py` file for the first time, you need to download the dataset (omitted in this repository due to its size) and generate the `.pt` file with all the images and labels.
To do so, just allocate some resources in the cluster (this code will require around 80GB of RAM) and run: 

```bash
source $(pwd)/federatedenv/bin/activate
python src/data_downloader.py
```

*Note*: If needed, adjust the settings in the [`settings.py`](./settings.py) file.

# 3. How to run

All the main loop of the federated learning is performed in the [`sbatch.sh`](./sbatch.sh) file. 
This is a bash script which will request resources from a cluster through the [SLURM](https://slurm.schedmd.com/) scheduler and perform sequentially the [`main.py`](./main.py) and [`aggregate_weights.py`](./src/aggregate_weights.py) files.

To run the code, you need to:

  - Set up the `setting.py` file with the desired settings
  - Launch the `sbatch.sh` file with the sbatch command
    ```bash
    sbatch sbatch.sh
    ```

# 4. Settings doc:

## 4.0 Neural Network settings

| **_Variable_**  | **_Type_** | **_Description_**                                                 |
|:---------------:|:----------:|:------------------------------------------------------------------|
|  `BATCH_SIZE`   |   `int`    | Batch size for the training                                       |
|   `N_EPOCHS`    |   `int`    | Number of epochs the single center will train the model           |
| `LEARNING_RATE` |  `float`   | Learning rate for the optimizer                                   |
|   `TRAINSIZE`   |  `float`   | Percentage of the local dataset each center will use for training |
|    `VALSIZE`    |  `float`   | Percentage of the local dataset each center will use for training |
|   `TESTSIZE`    |  `float`   | Percentage of the local dataset each center will use for training |

## 4.1 Data settings


|   **_Variable_**   |     **_Type_**      | **_Description_**                                                                                    |
|:------------------:|:-------------------:|:-----------------------------------------------------------------------------------------------------|
|    `DATA_PATH`     |      `string`       | Path in wich the `.pt` file with all the images and labels is stored                                 |
|   `RESULT_PATH`    |      `string`       | directory where to put all the results of the computations                                           |
|   `AUGMENT_DATA`   |       `bool`        | If `True` the data generated by the [`data_downloader.py`](src/data_downloader.py) will be augmented |
|   `DOWNLOAD_URL`   |      `string`       | URL from which to download the dataset is taken                                                      |
|     `ALLDATA`      |      `string`       | file name of the `.pt` file with all the images and labels                                           |
|     `ZIP_FILE`     |      `string`       | file name of the `.zip` file wile that will be downloaded from the `DOWNLOAD_URL`                    |
|   `EXTRACT_DIR`    |      `string`       | directory where to extract the `.zip` file                                                           |
|     `FILE_EXT`     |      `string`       | extension of the images that are going to be downloaded                                              |
| `PIC_SQUARE_SIZE`  |        `int`        | Size in which the images are going to be resized (equals height and width)                           |
| `LABELS`           | `list[string: int]` | List of the labels that are going to be used in the dataset                                          |
| `CLASS_SIZES`      | `list[int]`         | List of the number of images for each class in the dataset                                           |


## 4.2 Federated settings

| **_Variable_** | **_Type_** | **_Description_**                                                               |
|:--------------:|:----------:|:--------------------------------------------------------------------------------|
| `NITER_FED`    | `int`      | Number of iteration of the federated learning algorithm                         |
| `PERC`| `Tensor[NCENTERS, NCLASSES]` | Percentage of each class in the dataset of each of the centers         |
| `NCLASSES`     | `int`      | Number of classes in the dataset                                               ``` |
| `NCENTERS`     | `int`      | Number of centers that are going to be used in the federated learning algorithm |

## 4.3 Other settings

| **_Variable_** | **_Type_** | **_Description_**                                                                      |
|:--------------:|:----------:|:---------------------------------------------------------------------------------------|
| `VERBOSE`      | `bool`     | If `True` the code will print all the information about the training                   |
| `DEBUG`        | `bool`     | If `True` the code will print debug information during the execution                   |
| `PRINTWEIGHTS` | `bool`     | If `True` the code will print the weights of the network (used for debugging purposes) |

# 5. References:

***TODO:*** Insert references

