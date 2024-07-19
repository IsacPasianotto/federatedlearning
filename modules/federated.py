####
## Imports
####

# Downloaded Modules
import torch as th

import modules.networks as networks
from modules.traintest import test
from modules.dataset import build_Dataloader

import copy

def aggregate_models(models, trainSizes):
    """ Aggregate the weights of multiple models into a single model performing a weighted average of the weights.
    The weight is based on the size of each dataset

    Args:
        models (List[nn.Module]): List of PyTorch models
        trainSizes (List[int]): size of the datasets used to train each model

    Returns:
        nn.Module: A new model with the aggregated weights
    """
    total = sum(trainSizes)
    weights = (th.tensor(trainSizes, dtype=th.float) / total)

    # Extract state dictionaries from models
    state_dicts = [model.state_dict() for model in models]

    # Create a new model with the same architecture as the input models
    aggregated_model = copy.deepcopy(models[0])
    new_state_dict = aggregated_model.state_dict()

    for k in new_state_dict.keys():
        if 'num_batches_tracked' in k:
            # Special handling for BatchNorm layers
            new_state_dict[k] = state_dicts[0][k]
        else:
            stacked_tensors = th.stack([state_dict[k] for state_dict in state_dicts])
            new_state_dict[k] = (stacked_tensors * weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)

    # Load the aggregated weights into the new model
    aggregated_model.load_state_dict(new_state_dict)

    return aggregated_model


def evalAggregate(data, final_models, trainSizes):
    """ Evaluate the aggregated model on the test set

    Args:
        data (String): the file where to load data from
        final_models (List[nn.Module]): The results of the training process for each device
        trainSizes (List[int]): The sizes of the training datasets for each device
        batch_size (int): The batch size to be used
    """
    # Aggregate the weights of the two models to see the federated model
    federated_model = aggregate_models(final_models, trainSizes)

    for i, loader in enumerate(data):
        test(i, federated_model, loader, f'Accuracy of the federated model on the test set {i}: ')
    return federated_model
    
