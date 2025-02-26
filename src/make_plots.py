########
## Imports
########

import os
import re
import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("notebook")

RESULTS_PATH = '../results/'
PUT_TITLE = False
ACC_BASELINE=85.58

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
    STEP = 5
    MIN = 0
    MAX = 30
    # sort the to_read files by the simulation (search the "simulation" string in the path)
    for sim in simulations:    
        SETTINGS = RESULTS_PATH + sim + "/used_settings.py"
        N_ITER_FED = get_int_constant_from_file(SETTINGS, "N_ITER_FED") 
        N_EPOCHS = get_int_constant_from_file(SETTINGS, "N_EPOCHS")

        to_read: list[str] = find_target_files(RESULTS_PATH + sim, target)
        data = [th.load(d).reshape(N_ITER_FED, N_EPOCHS) for d in to_read if sim==d.split("/")[2]]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # for our use case, the global title is the path of the simulation
        title: str = sim.replace("_", " ").capitalize() + "\n" + target.replace("_", " ").capitalize() if PUT_TITLE else ""
        fig.suptitle(title, fontweight='bold', fontsize=18)
        indices = np.arange(MIN, MAX, STEP)
        ncols = 2 if len(indices) > 4 else 1
        for i, ax in enumerate(axs.flat):
            selected_data = data[i][indices]
            ax.plot(selected_data.T, label=[f"{j}" for j in indices])
            ax.set_title(f"Center {i}", fontweight='bold', fontsize=14)
            ax.set_xlabel("Epoch", fontweight='bold', fontsize=12)
            ax.set_ylabel("Loss", fontweight='bold', fontsize=12)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(12)
                label.set_fontweight("bold")

            ax.legend(title="Iteration", loc="best", ncol=ncols)
            ax.patch.set_facecolor(sns.axes_style()["axes.facecolor"])
            ax.patch.set_alpha(1)
        fig.tight_layout()
        # save the plot in the results folder
        
        #set the background color of the figure to be transparent
        fig.patch.set_alpha(0.0)

        plt.savefig(f"{RESULTS_PATH}/plots/{sim}_{target}.png", facecolor=fig.get_facecolor(), edgecolor='none')
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
        title = sim.replace("_", " ").capitalize() if PUT_TITLE else ""
        plot_single(test_accuracies_mean, aggregated_accuracies_mean, title, ax)
        fig.tight_layout()
        fig.patch.set_alpha(0.0)
        ax.patch.set_facecolor(sns.axes_style()["axes.facecolor"])

        plt.savefig(f"{RESULTS_PATH}/plots/{sim}_aggregated_by_center.png", facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        
def plot_accuracies(simulations):
    for sim in simulations:    
        test_accuracies = read_data(sim, "test_accuracies")
        aggr_accuracies = read_data(sim, "aggregated_accuracies")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # for our use case, the global title is the path of the simulation
        if PUT_TITLE:
            fig.suptitle(sim.replace("_", " ").capitalize(), fontweight='bold', fontsize=18)
        for i, ax in enumerate(axs.flat):
            plot_single(test_accuracies[i], aggr_accuracies[i], f"Center {i}", ax)
        fig.tight_layout()
        fig.patch.set_alpha(0.0)
        plt.savefig(f"{RESULTS_PATH}/plots/{sim}_accuracies.png", facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

def plot_single(test, aggregated, title, ax):
    ax.plot(test, label="Test")
    ax.plot(aggregated, label="Aggregated")
    # ax.set_ylim(70, 102)
    ax.set_ylim(65, ACC_BASELINE + 5)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel("Iteration", fontweight='bold', fontsize=12)
    ax.set_ylabel("Accuracy", fontweight='bold', fontsize=12)
    ax.tick_params(axis="both", which="major")

    # baseline Accuracy
    ax.axhline(y=ACC_BASELINE, color="r", linestyle="--", label="Baseline")

    # set bodl font to ticks
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight("bold")
    ax.legend(title="Type", loc="best")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.patch.set_facecolor(sns.axes_style()["axes.facecolor"])
    ax.patch.set_alpha(1)



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

    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_path) 
                                for f in filenames if target in f]
    sortedFiles = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return sortedFiles

def plot_baseline() -> None:
    """Plot baseline accuracy and losses in a unique file
    """
    base = RESULTS_PATH + 'baseline/' 
    accuracies = pd.read_csv(base + 'accuracies.csv')
    trlosses = pd.read_csv(base + 'train_losses.csv')
    vallosses = pd.read_csv(base + 'val_losses.csv')
    fig, axs = plt.subplots()
    axs2 = axs.twinx()
    axs.plot(trlosses, label='Train loss')
    axs.plot(vallosses, label='Val loss')
    axs2.plot(accuracies, label='Accuracy', color='green')
    axs.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axs.set_ylabel('Loss', fontsize=12, fontweight='bold')
    axs2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    for label in axs.get_yticklabels() + axs.get_xticklabels() + axs2.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    axs.set_xlim(0, 150)
    axs2.grid(False)
    # generate a single legend for both axes:
    lines, labels = axs.get_legend_handles_labels()
    lines2, labels2 = axs2.get_legend_handles_labels()
    axs2.legend(lines + lines2, labels + labels2, loc='best')
    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    plt.savefig(RESULTS_PATH + 'plots/baseline.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

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

    ## ---   Baseline  --- ##
    plot_baseline()

if __name__ == '__main__':
    main()

