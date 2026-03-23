import torch
import torch.nn as nn

from typing import List
from config.settings import NetworkSettings
from .layers import LinearBlock


class NetworkBody(nn.Module):
    """
    The network body consists of several layers that are shared by actor and critic.
    """
    def __init__(
        self, 
        netsettings: NetworkSettings
    ):
        """
        Inits NetworkBody.

        Args:
            netsettings (NetworkSettings): A tuple containing the configuration data.
        """
        super().__init__()
        self.input_dim = netsettings.num_observations
        self.hidden_dim = netsettings.hidden_dim
        self.num_layers = netsettings.num_layers
        self.use_memory = netsettings.use_memory

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        # Add the encoding layers
        for _ in range(self.num_layers):
            layers.append(
                LinearBlock(
                    input_dim = in_dim, 
                    output_dim = self.hidden_dim, 
                    weights_gain = netsettings.weights_gain
                )
            )
            in_dim = self.hidden_dim

        # Optional: Add the memory layer
        # TODO

        # Convert the list into a sequence of nn modules
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes observations through the shared network body.

        Args:
            x (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: Encoded feature tensor.
        """
        return self.body(x)
