import torch
import torch.nn as nn

from typing import List, Tuple, Optional
from config.settings import NetworkSettings
from .init_memory import MemModuleCreator
from .layers import LinearBlock


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
        super(NetworkBody, self).__init__()
        self.net_settings = netsettings

        # Add the encoding layers
        layers: List[nn.Module] = []
        in_dim = self.net_settings.num_observations
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
        self.memory_block: Optional[nn.Module] = None
        if self.net_settings.memory is not None:
            memory_settings = self.net_settings.memory_settings
            self.memory_block = MemModuleCreator.create(
                self.net_settings.memory, self.net_settings.hidden_dim, memory_settings
            )

        # Convert the list into a sequence of nn modules
        self.body = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes observations through the shared network body.

        Args:
            x (torch.Tensor): Input observation tensor.
            memory (Optional[torch.Tensor]): Optional recurrent memory input.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Encoded features and optional updated recurrent memory.
        """
        x = self.body(x)

        # Return directly x if the memory module is not used
        if self.memory_block is None:
            return x, None

        # Recurrent modules consume sequence inputs [B, T, F].
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Delegate memory tensor construction to the selected recurrent block.
        if memory is None:
            memory = self.memory_block.init_memory(
                batch_size=x.size(0), device=x.device, dtype=x.dtype
            )

        return self.memory_block(x, memory)
