import torch
import torch.nn as nn

from typing import List, Tuple, Optional
from config.settings import NetworkSettings
from .layers import CfCBlock, LinearBlock, LSTMBlock


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
        self.memory_settings = self.net_settings.memory
        self.memory_block: Optional[nn.Module] = None
        if self.memory_settings is not None:
            if self.memory_settings.module_type == "lstm":
                self.memory_block = LSTMBlock(
                    input_dim=self.net_settings.hidden_dim,
                    hidden_dim=self.memory_settings.memory_dim,
                )
            elif self.memory_settings.module_type == "cfc":
                self.memory_block = CfCBlock(
                    input_dim=self.net_settings.hidden_dim,
                    cfc_units=self.memory_settings.memory_dim,
                )
            else:
                raise ValueError(
                    f"Expected LSTM or CfC, got {self.memory_settings.module_type}"
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

        if not isinstance(self.memory_block, (LSTMBlock, CfCBlock)):
            raise TypeError(
                f"Expected LSTMBlock or CfCBlock, got {type(self.memory_block).__name__}"
            )

        # Recurrent modules consume sequence inputs [B, T, F].
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Delegate memory tensor construction to the selected recurrent block.
        if memory is None:
            memory = self.memory_block.init_memory(
                batch_size=x.size(0), device=x.device, dtype=x.dtype
            )

        return self.memory_block(x, memory)
