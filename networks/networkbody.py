import torch
import torch.nn as nn

from typing import List, Tuple, Optional
from config.settings import NetworkSettings
from .layers import LinearBlock, LSTMBlock


class NetworkBody(nn.Module):
    """
    The network body consists of several layers that are shared by actor and critic.
    """

    def __init__(self, netsettings: NetworkSettings):
        """
        Inits NetworkBody.

        Args:
            netsettings (NetworkSettings): A tuple containing the configuration data.
        """
        super().__init__()
        self.net_settings = netsettings

        layers: List[nn.Module] = []
        in_dim = self.net_settings.num_observations
        # Add the encoding layers
        for _ in range(self.net_settings.num_layers):
            layers.append(
                LinearBlock(
                    input_dim=in_dim,
                    output_dim=self.net_settings.hidden_dim,
                    weights_gain=self.net_settings.weights_gain,
                )
            )
            in_dim = self.net_settings.hidden_dim

        # Add the memory layer
        if self.net_settings.memory is not None:
            memory = self.net_settings.memory
            if memory.module_type == "lstm":
                self.memory = LSTMBlock(
                    input_dim=self.net_settings.hidden_dim,
                    hidden_dim=memory.memory_dim,
                )
            elif memory.module_type == "cfc":
                pass

        # Convert the list into a sequence of nn modules
        self.body = nn.Sequential(*layers)

    def forward(
        self, 
        x: torch.Tensor,
        memory: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes observations through the shared network body.

        Args:
            x (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: Encoded feature tensor.
        """
        x = self.body(x)

        # Return directly x if the memory module is not used
        if self.net_settings.memory is None:
            return x, None

        # Unsqueeze the input tensor for time axis
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Initialize memory if it is None
        if memory is None:
            batch_size = x.size(0)
            num_layers = 1
            hidden_dim = self.net_settings.memory.memory_dim

            h0 = torch.zeros(num_layers, batch_size, hidden_dim)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim)
            memory = torch.cat((h0, c0), dim=-1)

        return self.memory(x, memory)
