########
## Imports
########

import os
import sys
import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_EPOCHS = 50
N_ITER_FED = 30
RESULTS_PATH = '../results/'

########
## functions
########

def plot_losses_same_network(
        ax:     plt.Axes,
        losses: list[list[float]],
        title:  str,
        max:    int = N_ITER_FED,
        step:   int = 4
    ) -> None:
    """ Plot all the losses of the same network during different iterations of
    the federated learning process

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the losses on
    losses : list[list[float]]
        The losses of the network during the iterations
    title : str
        The title of the plot
    max : int
        The maximum number of iterations (if the convergence is reached before, could make
        the plot more readable)
    step : int
        The step between two iterations to plot
    """

    for i in range(max):
        if i % step == 0:
            ax.plot(losses[i], label=f"{i}")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(title="Iterations", loc="best")

def plot_losses_same_network_gird(
        all_losses:    list[list[list[float]]],
        figsize:       tuple[int, int] = (15, 10),
        max:           int = 21,
        step:          int = 4,
        global_title:  str = "Losses of the same network during the iterations",
        custom_titles: list[str] = None,
        suffix:        str = ""
    )  -> None:
    """ Plot all the losses of the train of all the centers during the processes og the
    federated learning.  Remark: this fucntion is thought to be used for the conducted experiments
    in which we considered always 4 centers.

    Parameters
    ----------
    all_losses : list[list[list[float]]]
        The losses of the networks during the iterations
    figsize : tuple[int, int]
        The size of the figure
    max : int
        The maximum number of iterations (if the convergence is reached before, could make
        the plot more readable)
    step : int
        The step between two iterations to plot
    global_title : str
        The title of the plot
    custom_titles : list[str]
        The titles of the subplots
    suffix : str
        The suffix to add to the plot file name
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # for our use case, the global title is the path of the simulation
    title: str = global_title.split("/")[-1]
    title = " ".join([w.capitalize() for w in title.split("_")])

    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axs.flat):
        t = custom_titles[i] if custom_titles is not None else f"Center {i}"
        plot_losses_same_network(ax, all_losses[i], t, max, step)

    plot_file_name= global_title.split("/")[-1] + suffix
    # plt.show()
    # save the plot in the results folder
    plt.savefig(f"{RESULTS_PATH}/plots/{plot_file_name}.png")
    plt.close()

def plot_accuracies(
        ax: plt.Axes,
        train_accuracies: np.array,
        aggregated_accuracies: np.array,
        title: str,
        suffix: str = ""
    ) -> None:
    """ Plot the accuracies of the train and the aggregated accuracies

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot the accuracies on
    train_accuracies : np.array
        The accuracies of the train
    aggregated_accuracies : np.array
        The aggregated accuracies
    title : str
        The title of the plot
    suffix : str
        The suffix to add to the plot file name
    """
    ax.plot(train_accuracies, label="Train")
    ax.plot(aggregated_accuracies, label="Aggregated")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Type", loc="best")
    title = title.split("/")[-1]
    title = " ".join([w.capitalize() for w in title.split("_")])
    plot_file_name = title + suffix
    plt.savefig(f"{RESULTS_PATH}/plots/{plot_file_name}.png")
    plt.close()

def find_target_file(path: str, target: str) -> list[str]:
    """ Find all the files in the path that contain the target string

    Parameters
    ----------
    path : str
        The path in which to search the files
    target : str
        The string to search in the files

    Returns
    -------
    list[str]
        The list of all the files that contain the target string
    """
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if target in f]

def load_data_torch (files_to_load: list[str]) -> list[th.Tensor]:
    """ Load all the tensors in the list of files

    Parameters
    ----------
    files_to_load : list[str]
        The list of files to load

    Returns
    -------
    list[th.Tensor]
        The list of all the tensors loaded
    """
    return [th.load(f).reshape(N_ITER_FED, N_EPOCHS) for f in files_to_load]

def load_data_pandas (files_to_load: list[str]) -> np.array:
    """ Load all the accuracies in the list of files

    Parameters
    ----------
    files_to_load : list[str]
        The list of files to load

    Returns
    -------
    np.array
        The array of all the accuracies loaded
    """
    return np.array([pd.read_csv(f).values for f in files_to_load]).squeeze()


def main() -> None:

    # create a folder to save the plots in the results folder
    # if it already exists, remove the previous plots
    os.makedirs(f"{RESULTS_PATH}/plots", exist_ok=True)
    for f in os.listdir(f"{RESULTS_PATH}/plots"):
        os.remove(f"{RESULTS_PATH}/plots/{f}")

    # Get all the situations in which the federated learning was conducted (all the folders in the results folder)
    simulations: list[str] = [f[0] for f in os.walk(RESULTS_PATH)]
    # remove the first element that is the results folder itself, plus the "plots" and "__pycache__" folders
    simulations = simulations[1:]
    simulations = [s for s in simulations if "plots" not in s and "__pycache__" not in s]

    ## ---   Losses  --- ##

    target: str = "train_losses"
    to_read: list[str] = find_target_file(RESULTS_PATH, target)

    # sort the to_read files by the simulation (serach the "simulation" string in the path)
    to_read_clusterized: dict[str, list[str]] = {}
    for s in simulations:
        to_read_clusterized[s] = [f for f in to_read if s in f]

    read: dict[str, list[th.Tensor]] = {}
    for k,v in to_read_clusterized.items():
        read[k] = load_data_torch(v)

    for k,v in read.items():
        plot_losses_same_network_gird(v, global_title=k, suffix="_train_losses")

    # ----
    target: str = "val_losses"
    to_read: list[str] = find_target_file(RESULTS_PATH, target)
    to_read_clusterized: dict[str, list[str]] = {}
    for s in simulations:
        to_read_clusterized[s] = [f for f in to_read if s in f]
    read: dict[str, list[th.Tensor]] = {}
    for k,v in to_read_clusterized.items():
        read[k] = load_data_torch(v)
    for k,v in read.items():
        plot_losses_same_network_gird(v, global_title=k, suffix="_val_losses")

    ## ---   Accuracies  --- ##

    target: str = "aggregated_accuracies"
    to_read: list[str] = find_target_file(RESULTS_PATH, target)
    to_read_clusterized: dict[str, list[str]] = {}
    for s in simulations:
        to_read_clusterized[s] = [f for f in to_read if s in f]

    read: dict[str, np.ndarray] = {}
    for k,v in to_read_clusterized.items():
        read[k] = load_data_pandas(v)

    aggregated_accuracies: dict[str, np.array] = read

    aggregated_accuracies_mean: dict[str, np.array] = {}
    for k,v in read.items():
        aggregated_accuracies_mean[k] = np.mean(v, axis=0)

    target: str = "test_accuracies"
    to_read: list[str] = find_target_file(RESULTS_PATH, target)
    to_read_clusterized: dict[str, list[str]] = {}
    for s in simulations:
        to_read_clusterized[s] = [f for f in to_read if s in f]

    read: dict[str, np.ndarray] = {}
    for k,v in to_read_clusterized.items():
        read[k] = load_data_pandas(v)

    test_accuracies: dict[str, np.array] = read
    test_accuracies_mean: dict[str, np.array] = {}
    for k,v in read.items():
        test_accuracies_mean[k] = np.mean(v, axis=0)

    for k in aggregated_accuracies.keys():
        fig, ax = plt.subplots()
        plot_accuracies(ax, aggregated_accuracies_mean[k], test_accuracies_mean[k], k, suffix="_aggregated_by_center")

if __name__ == '__main__':
    main()