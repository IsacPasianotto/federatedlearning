########
## Imports
########

import os
import re
import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = '../results/'

########
## functions
########

def get_int_constant_from_file(file_path, constant_name):
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(rf'{constant_name}:\s*int\s*=\s*(\d+)\s*$', line)
            if match:
                return int(match.group(1))
    return None  # Return None if constant not found

    
def read_data(sim, target):
    to_read: list[str] = find_target_files(RESULTS_PATH + sim, target)
    return np.array([
        pd.read_csv(f, header=None).values.ravel()
        for f in to_read
        if sim == f.split("/")[2]
    ])
     


def plot_losses(simulations, target):
    STEP = 4
    MAX = 21
    # sort the to_read files by the simulation (search the "simulation" string in the path)
    for sim in simulations:    
        SETTINGS = RESULTS_PATH + sim + "/used_settings.py"
        N_ITER_FED = get_int_constant_from_file(SETTINGS, "NITER_FED")  # TO CHANGE NITER_FED -> N_ITER_FED #############################
        N_EPOCHS = get_int_constant_from_file(SETTINGS, "N_EPOCHS")
        to_read: list[str] = find_target_files(RESULTS_PATH + sim, target)
        data = [th.load(d).reshape(N_ITER_FED, N_EPOCHS) for d in to_read if sim==d.split("/")[2]]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # for our use case, the global title is the path of the simulation
        title: str = sim.replace("_", " ").capitalize() + "\n" + target.replace("_", " ").capitalize()
        fig.suptitle(title, fontsize=16)
        for i, ax in enumerate(axs.flat):
            indices = np.arange(0, MAX, STEP)
            selected_data = data[i][indices]
            ax.plot(selected_data.T, label=[f"{j}" for j in indices])
            ax.set_title(f"Center {i}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(title="Iterations", loc="best", ncol=2)
    
        # save the plot in the results folder
        plt.savefig(f"{RESULTS_PATH}/plots/{sim}_{target}.png")
        plt.close()
    
def plot_avg_accuracies(
        simulations: str
    ) -> None:
    """ Plot the accuracies of the train and the aggregated accuracies

    Parameters
    ----------
    simulation : str
        The simulation to plot the accuracies for
    """
    for sim in simulations:
        test_accuracies_mean       = np.mean(read_data(sim, "test_accuracies"), axis=0)
        aggregated_accuracies_mean = np.mean(read_data(sim, "aggregated_accuracies"), axis=0)
        fig, ax = plt.subplots()  
        plot_single(test_accuracies_mean, aggregated_accuracies_mean, sim.replace("_", " ").capitalize(), ax)
        plt.savefig(f"{RESULTS_PATH}/plots/{sim}_aggregated_by_center.png")
        plt.close()
        
def plot_accuracies(simulations):
    for sim in simulations:    
        test_accuracies = read_data(sim, "test_accuracies")
        aggr_accuracies = read_data(sim, "aggregated_accuracies")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # for our use case, the global title is the path of the simulation
        fig.suptitle(sim.replace("_", " ").capitalize(), fontsize=16)
        for i, ax in enumerate(axs.flat):
            plot_single(test_accuracies[i], aggr_accuracies[i], f"Center {i}", ax)
        # save the plot in the results folder
        plt.savefig(f"{RESULTS_PATH}/plots/{sim}_accuracies.png")
        plt.close()

def plot_single(test, aggregated, title, ax):
    ax.plot(test, label="Test")
    ax.plot(aggregated, label="Aggregated")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Type", loc="best")


def find_target_files(base_path: str, target: str) -> list[str]:
    """ Find all the files in the path that contain the target string

    Parameters
    ----------
    base_path : str
        The path in which to search the files
    target : str
        The string to search in the files

    Returns
    -------
    list[str]
        The list of all the files that contain the target string
    """
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_path) 
                                for f in filenames if target in f]


def main() -> None:
    # create a folder to save the plots in the results folder
    # if it already exists, remove the previous plots
    os.makedirs(f"{RESULTS_PATH}/plots", exist_ok=True)
    for f in os.listdir(f"{RESULTS_PATH}/plots"):
        os.remove(f"{RESULTS_PATH}/plots/{f}")

    # Get all the situations in which the federated learning was conducted (all the folders in the results folder)
    _, d, _ = next(os.walk(RESULTS_PATH))
    simulations = [f for f in d if "balanced" in f]
    ## ---   Losses  --- ##
    plot_losses(simulations, "train_losses")
    plot_losses(simulations, "val_losses")

    ## ---   Accuracies  --- ##
    plot_avg_accuracies(simulations)
    plot_accuracies(simulations)


if __name__ == '__main__':
    main()
