# Federated learning for brain cancer classification

<p style="text-align: center;">

[![License](https://img.shields.io/badge/License-MIT%20license-mitlicense?style=for-the-badge&logo=MIT&color=%23FF9E0F)](https://github.com/IsacPasianotto/federatedlearning/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-Python?style=for-the-badge&logo=Python&logoColor=%23FECC00&color=%233776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-PyTorch?style=for-the-badge&logo=PyTorch&logoColor=white&color=%23EE4C2C)](https://pytorch.org/)

</p>

_Authors_:\
&emsp;[**Pasianotto Isac**](https://github.com/IsacPasianotto/)\
&emsp;[**Rossi Davide**](https://github.com/DavideRossi1/)


>> ***Warning:***
>>
>> This repository contains in its history some large files (saved weights of the neural network) that can significantly slow down the cloning process and are unnecessary for the code execution.
>> 
>> For this reason, it is strongly recommended to clone the clone-dedicated branch with the following command:
>>
>> ```bash
>>    git clone --depth 1 https://github.com/IsacPasianotto/federatedlearning.git --branch fastdownload --single-branch
>> ```

# 0. Table of contents

- [Federated learning for brain cancer classification](#federated-learning-for-brain-cancer-classification)
- [0. Table of contents](#0-table-of-contents)
- [1. Introduction](#1-introduction)
  - [1.0 The idea behind federated learning](#10-the-idea-behind-federated-learning)
  - [1.1 The dataset](#11-the-dataset)
  - [1.2 The model and the training](#12-the-model-and-the-training)
- [2. Getting started](#2-getting-started)
  - [2.0 Prerequisites](#20-prerequisites)
  - [2.1 Generate the dataset:](#21-generate-the-dataset)
- [3. How to run](#3-how-to-run)
- [4. Settings doc](#4-settings-doc)
  - [4.0 Neural Network settings](#40-neural-network-settings)
  - [4.1 Data settings](#41-data-settings)
  - [4.2 Federated learning settings](#42-federated-learning-settings)
  - [4.3 Other settings](#43-other-settings)

# 1. Introduction

## 1.0 The idea behind federated learning

Federated learning is a machine learning approach that allows multiple entities to collaboratively train a model without sharing their data. This is not only useful to speedup the training process (since it can be therefore carried on in a distributed scenario), but mainly to protect the privacy of the data by avoiding data sharing. A tipical scenario in which federated learning shows its potential is the medical field, where the data is sensitive and the privacy of the patients must be preserved: for example, different hospitals, each with its own dataset, can collaborate to train a model that can be used to predict a certain disease, without sharing the data used for the training.

Going more in detail, the federated learning algorithm works as follows:

1. All centers (hospitals) have their own dataset, which is not shared with the others.
2. All centers agree on a model architecture and a loss function and train this same model on their own dataset.
3. The centers send the weights of the learned model to a central server, which aggregates them in some way (e.g. by averaging them).
4. The central server sends the aggregated weights back to the centers, which start again the training process with the new weights.

This process is repeated for a certain number of iterations, and at the end, the central server has a model that has been trained on all the datasets, without actually having access to the data itself.

This approach has several advantages and can be performed on several different scenarios, not necessarily involving similar datasets. For example:
- datasets can have different **size**: some centers may be larger than other, therefore having more data to train the model
- datasets can have different **distributions**: some centers may be specialized on a certain type of disease, therefore having many more samples of that class

## 1.1 The dataset

In this repository, we present a federated learning algorithm applied to the classification of brain' Magnetic Resonance images to diagnose some typologies of cancer. The dataset used is the [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download), openly available on Kaggle. The dataset contains ~7000 images of brain MRI, divided in 4 more or less equally represented classes: glioma, meningioma, pituitary and no tumor. The goal is to develop a model to classify these images, split the dataset among multiple different centers, let them train the model on their own data, aggregate the weights to obtain a final model and evaluate its performance.

## 1.2 The model and the training

The model used in this repository is a simple Convolutional Neural Network with 3 convolutional layers and 4 fully connected layers. Adam optimizer and cross entropy loss are used to train the model.

The model has been trained on multi-node, multi-GPU clusters (Leonardo and Orfeo), using the PyTorch library and `torchrun`: each center uses a different GPU to train the model, and the central server aggregates the weights of the model at the end of each iteration.

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
To do so, just allocate some resources in the cluster (this code will require around 160GB of RAM) and run:

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
  - Fill the SLURM directives in the `sbatch.sh` file accordingly to the cluster you are using
  - Launch the `sbatch.sh` file with the sbatch command
    ```bash
    sbatch sbatch.sh
    ```

# 4. Settings doc

## 4.0 Neural Network settings

| **_Variable_**  | **_Type_** | **_Description_**                                                   |
|:---------------:|:----------:|:--------------------------------------------------------------------|
| `BATCH_SIZE`    | `int`      | Batch size for the training                                         |
| `N_EPOCHS`      | `int`      | Number of epochs the single center will train the model             |
| `LEARNING_RATE` | `float`    | Learning rate for the optimizer                                     |
| `WEIGHT_DECAY`  | `float`    | Weight decay for the optimizer                                      |
| `TRAIN_SIZE`    | `float`    | Percentage of the local dataset each center will use for training   |
| `VAL_SIZE`      | `float`    | Percentage of the local dataset each center will use for validation |
| `TEST_SIZE`     | `float`    | Percentage of the local dataset each center will use for test       |

## 4.1 Data settings

| **_Variable_**  | **_Type_**          | **_Description_**                                                                                     |
|:---------------:|:-------------------:|:------------------------------------------------------------------------------------------------------|
| `DATA_PATH`     | `string`            | Path in wich the `.pt` file with all the images and labels is stored                                  |
| `RESULTS_PATH`  | `string`            | Directory where to put all the results of the computations                                            |
| `BASELINE_PATH` | `string`            | Directory where to put the results of the baseline (network trained on the whole data) results        |
| `AUGMENT_DATA`  | `bool`              | If `True`, the data generated by the [`data_downloader.py`](src/data_downloader.py) will be augmented |
| `DOWNLOAD_URL`  | `string`            | URL from which to download the dataset                                                                |
| `ALL_DATA`      | `string`            | File name of the `.pt` file with all the images and labels                                            |
| `ZIP_FILE`      | `string`            | File name of the `.zip` file wile that will be downloaded from the `DOWNLOAD_URL`                     |
| `EXTRACT_DIR`   | `string`            | Directory where to extract the `.zip` file                                                            |
| `FILE_EXT`      | `string`            | Extension of the images that are going to be downloaded                                               |
| `PIC_SIZE`      | `int`               | Size in which the images are going to be resized (square images)                                      |
| `LABELS`        | `list[string: int]` | List of the labels that are going to be used in the dataset                                           |



## 4.2 Federated learning settings

| **_Variable_** | **_Type_**                   | **_Description_**                                                               |
|:--------------:|:----------------------------:|:--------------------------------------------------------------------------------|
| `N_ITER_FED`   | `int`                        | Number of iterations of the federated learning algorithm                        |
| `PERC`         | `Tensor[NCENTERS, NCLASSES]` | Percentage of each class of the dataset for each center                         |
| `N_CLASSES`    | `int`                        | Number of classes in the dataset                                                |
| `N_CENTERS`    | `int`                        | Number of centers that are going to be used in the federated learning algorithm |
| `CLASS_SIZES`  | `list[int]`                  | List of the number of images for each class in the dataset                      |

## 4.3 Other settings

| **_Variable_**  | **_Type_** | **_Description_**                                                                      |
|:---------------:|:----------:|:---------------------------------------------------------------------------------------|
| `VERBOSE`       | `bool`     | If `True` the code will print all the information about the training                   |
| `DEBUG`         | `bool`     | If `True` the code will print debug information during the execution                   |
| `PRINT_WEIGHTS` | `bool`     | If `True` the code will print the weights of the network (used for debugging purposes) |

