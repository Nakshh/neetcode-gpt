import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sd = math.sqrt(2/(fan_in + fan_out))
        output = torch.round(torch.randn(fan_out, fan_in) * sd, decimals = 4).tolist()
        return output

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        sd = math.sqrt(2/fan_in)
        output = torch.round(torch.randn(fan_out, fan_in) * sd, decimals = 4).tolist()
        return output


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        # activation = torch.randn(input_dim)
        weights = []
        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim
            fan_out = hidden_dim
            
            sd = 1.0    
            if init_type == "xavier":
                sd = math.sqrt(2/(fan_in + fan_out))
            elif init_type == "kaiming":
                sd = math.sqrt(2/fan_in)
            
            w = torch.randn(fan_out, fan_in) * sd
            weights.append(w)
        
        activation = torch.randn(input_dim)
        stds = []
        for weight in weights:
            activation = weight @ activation
            activation = torch.relu(activation)
            stds.append(round(activation.std().item(), 2))

        return stds

