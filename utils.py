import numpy as np
import random
import torch

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility in Python, NumPy, and PyTorch.
    This function also configures CUDA settings to ensure deterministic behavior.
    
    Parameters:
    seed (int): The seed value to set for all libraries.
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # For CUDA (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Ensuring deterministic behavior in PyTorch (optional but helpful for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Calculate SAIDI for each city based on predicted or actual outages
def calc_SAIDI(customer, outage):
    """
    Calculate SAIDI for each city based on customer numbers and outage data over the whole time period.

    Args:
        customer (np.array): Array of total customer counts for each city.
        outage (np.array): 2D Array of outage values for each city across all time steps.

    Returns:
        np.array: Array of SAIDI values for each city.
    """
    SAIDI = []
    for i in range(outage.shape[0]):
        saidi = np.sum(outage[i]) / customer[i]
        SAIDI.append(saidi)
    return np.array(SAIDI)
