import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.helpers import *
from utils.config import get_checkpoint_paths
#extract_hyperparameters, update_config,
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from utils.metrics import calculate_metrics

def weighted_mse_loss(predictions, targets, weights):
    """
    Compute the weighted MSE loss.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.
        weights (torch.Tensor): Weights corresponding to each prediction (based on occupancy).

    Returns:
        loss (torch.Tensor): The weighted MSE loss.
    """
    # Calculate the MSE loss
    mse = (predictions - targets) ** 2

    # Ensure weights shape matches the shape of mse
    if weights.dim() == 1:
        # If weights is 1D, make it match the dimension of mse
        weights = weights.unsqueeze(1).expand_as(mse)

    # Apply weights
    weighted_mse = mse * weights

    # Return the mean of the weighted MSE
    return weighted_mse.mean()

def homogeneous_weighted_mse_loss(predictions, targets, weights, epsilon=1e-6):
    """
    Compute the weighted MSE loss where the targets are scaled by the last entry of the predictions.

    Args:
        predictions (torch.Tensor): Model predictions (homogeneous coordinates).
        targets (torch.Tensor): Ground truth values.
        weights (torch.Tensor): Weights corresponding to each prediction (based on occupancy).
        epsilon (float): Small value to avoid division by zero (default: 1e-6).

    Returns:
        loss (torch.Tensor): The weighted MSE loss.
    """

    # Multiply the targets by the last entry of the predictions
    scaling_factors = predictions[:, -1].unsqueeze(1) + epsilon
    targets[:, :-1] *= scaling_factors

    # Calculate the MSE loss
    mse = (predictions - targets) ** 2

    # Ensure weights shape matches the shape of mse
    if weights.dim() == 1:
        # If weights is 1D, make it match the dimension of mse
        weights = weights.unsqueeze(1).expand_as(mse)

    # Apply weights
    weighted_mse = mse * weights

    # Return the mean of the weighted MSE
    return weighted_mse.mean()

def create_model_from_config(config: dict, checkpoint_name: str = None, load_checkpoint: bool = False):
    """
    Create a model from the given configuration, with an option to load the state from a checkpoint.

    Args:
        config (dict): The configuration dictionary.
        checkpoint_name (str, optional): Name of the checkpoint file. If None, it checks `config['model']['checkpoint_name']`.
        load_checkpoint (bool): Whether to load the model weights from the checkpoint. Default is False.

    Returns:
        model (torch.nn.Module): The initialized model.
        state (dict): The loaded state if a checkpoint is provided and loaded, else None.
    """
    # Create the model
    model_config = config['model']
    if 'state' in model_config: model_config.pop('state')
    model_type = model_config['type']  # Get model type from the config
    model_class = globals()[model_type]  # Dynamically get the model class
    model = model_class(config)  # Initialize the model

    state = None  # Default state if no checkpoint is loaded
    # Load the checkpoint only if requested
    if load_checkpoint and checkpoint_name is not None:
        checkpoint_path = get_checkpoint_paths(config, checkpoint_name)
        model, state = load_checkpoint_if_exists(config, model, 'cpu', checkpoint_path)
        #model, state = load_checkpoint_into_model(config, model, checkpoint_name)

    return model, state  # Return the model and state (if loaded)



# def load_checkpoint_into_model(config: dict, model: torch.nn.Module, checkpoint_name: str = None):
#     """
#     Load the model state from a checkpoint into the provided model.
#
#     Args:
#         config (dict): The configuration dictionary containing paths information.
#         model (torch.nn.Module): The model instance to load the checkpoint into.
#         checkpoint_name (str, optional): The name of the checkpoint file. If None, use `config['model']['checkpoint_name']`.
#
#     Returns:
#         dict: The loaded state from the checkpoint.
#     """
#     # Get checkpoint path using the helper function
#     checkpoint_path = get_checkpoint_paths(config, checkpoint_name)
#
#     # Check if the checkpoint file exists
#     if not os.path.isfile(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")
#
#     # Load the checkpoint state
#     state = torch.load(checkpoint_path)
#
#     # Load the state dictionary into the model
#     model.load_state_dict(state['net'])
#
#     return model, state

def save_checkpoint(checkpoint_path: str, model_state: dict, best_epoch: int = 0, epoch: int = 0, best_val_loss: float = np.inf, optimizer_state: dict = {}): #, scalers: dict = {}):
    """
    Save a model checkpoint to a file.

    Args:
        checkpoint_path (str): Path where the checkpoint file will be saved.
        best_epoch (int): The epoch number that achieved the best validation performance.
        epoch (int): The current epoch number at the time of saving the checkpoint.
        best_val_loss (float): The lowest validation loss recorded up to this point.
        model_state (dict): State dictionary of the model, containing parameters and buffers.
        optimizer_state (dict): State dictionary of the optimizer, containing state and parameter information.
        scalers (dict, optional): A dictionary containing scalers for features and targets, used to scale data during training.
    """
    torch.save({
        'best_epoch': best_epoch,  # Best epoch with the lowest validation loss
        'last_epoch': epoch,  # Current epoch number
        'best_val_loss': best_val_loss,  # Best validation loss recorded
        'net': model_state,  # Model state dictionary (weights, biases, etc.)
        'optimizer': optimizer_state  # Optimizer state dictionary (state, learning rates, etc.)
        #'scalers': scalers  # Optional dictionary with feature and target scalers
    }, checkpoint_path)  # Save all the information to the checkpoint file


def update_checkpoint_epoch(checkpoint_path: str, epoch: int, device='cpu'):
    """
    Update the epoch number in an existing checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        epoch (int): The new epoch number to update.
        device (str): Device where the model is located, defaults to 'cpu'.
    """
    state = torch.load(checkpoint_path, map_location=device)  # Load the checkpoint
    state['last_epoch'] = epoch  # Update the last_epoch field
    torch.save(state, checkpoint_path)  # Save the updated checkpoint


import os



def load_checkpoint_if_exists(config, model, device, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.pop("net"))
        #optimizer.load_state_dict(state.get("optimizer", {}))
        config['model']['state'] = state
        return model, state
    return model, config['model'].get('state', None)

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


