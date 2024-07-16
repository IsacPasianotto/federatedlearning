####
## Imports
####

# Downloaded Modules
import torch as th

import modules.networks as networks
from modules.traintest import test
from modules.dataset import build_Dataloader

def aggregate_weights(state_dicts, trainSizes):
    """ Aggregate the weights of multiple models into a single model performing a weighted average of the weights.
    The weighted is based on the size of each dataset

    Args:
        state_dicts (List[dict]): state_dicts of the models
        trainSizes (List[int]): size of the datasets used to train each model

    Returns:
        Dict: the dictionary with the weighted sum of the weights
    """
    total = sum(trainSizes)
    weights = (th.tensor(trainSizes, dtype=th.float) / total) # weights for each dataset, based on the size of the dataset, reshaped to be broadcastable
    #aggregate the weights
    new_state_dict = {}
    for k in state_dicts[0].keys():  # keys are the names the network nodes weights, biases, etc --> they are the same for all the networks since they are all identical
        stacked_tensors = th.stack([state_dict[k] for state_dict in state_dicts])
        new_state_dict[k] = (stacked_tensors * weights.view(-1, *([1] * (stacked_tensors.dim() - 1)))).sum(dim=0)
        # explanation of the line above:
        # [1] * (stacked_tensors.dim() - 1) creates a list of 1s with the same length as the dimensions of the stacked_tensors minus 1
        # -1 is used to keep the dimensions of the weights tensor the same as the stacked_tensors tensor, with the first dimension inferred from the weights tensor
        # the weights tensor is reshaped to be broadcastable to the stacked_tensors tensor
        # note that state_dict[k] could be a bias or a weight, which have different dimensions
    return new_state_dict


def evalAggregate(data, results, trainSizes, batch_size):
    """ Evaluate the aggregated model on the test set

    Args:
        data (String): the file where to load data from
        results (List[Dict]): The results of the training process for each device
        trainSizes (List[int]): The sizes of the training datasets for each device
        batch_size (int): The batch size to be used
    """
    # Aggregate the weights of the two models to see the federated model
    federated_model = networks.BrainClassifier()
    federated_weights = aggregate_weights([results[i]['state_dict'] for i in range(len(results))], trainSizes)
    federated_model.load_state_dict(federated_weights)

    data_agg = th.load(data)
    data_agg_train, data_agg_val, data_agg_test = data_agg.train_val_test_split()
    # train_loader_agg = build_Dataloader(data_agg_train, batch_size)
    # val_loader_agg   = build_Dataloader(data_agg_val, batch_size)
    test_loader_agg    = build_Dataloader(data_agg_test, batch_size)
    test(0, federated_model, test_loader_agg, f'Accuracy of {len(results)} aggregated models on test set: ')
 
