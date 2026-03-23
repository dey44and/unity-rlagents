from .init_strategy import InitializationEnum, InitStrategy, LinearInitStrategy, LSTMInitStrategy, CfCInitStrategy
from ncps.torch import CfC
from typing import Tuple

import torch
import torch.nn as nn


# Unlike MLAgents, I decided to use orthogonal init, instead of Xavier/Kaiming init.
class LinearBlock(nn.Module):
    """
    The linear block consists of a dense layer, a normalization layer, and a SiLU activation layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weights_init: InitializationEnum = InitializationEnum.ORTHOGONAL,
        weights_gain: float = 1.0,
        bias_init: InitializationEnum = InitializationEnum.ZEROS,
    ):
        """
        Inits the linear block.

        Args:
            input_dim (int): The size of the input tensor.
            output_dim (int): The size of the output tensor.
            weights_init (InitializationEnum): The method used for weights initialization.
            weights_gain (float): The gain should be applied to weights.
            bias_init (InitializationEnum): The method used for bias initialization.
        """
        super(LinearBlock, self).__init__()
        # Configure the linear layer, and initialize weights
        linear = nn.Linear(input_dim, output_dim, True)
        initializer: InitStrategy = LinearInitStrategy(
            weights_init, weights_gain, bias_init
        )
        initializer.initialize(linear)

        # Configure the normalization + activation
        norm = nn.LayerNorm(output_dim)
        activation = nn.SiLU()

        # Build the linear block
        self.linear_block = nn.Sequential(linear, norm, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_block(x)


# Here, I also decided to separate the initialization technique between forward/recurrent weights.
class LSTMBlock(nn.Module):
    """
    Recurrent block wrapping an LSTM layer and memory state handling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        batch_first: bool = True,
        forget_bias: float = 1.0,
        weights_init: InitializationEnum = InitializationEnum.XAVIER_UNIFORM,
        recurrent_init: InitializationEnum = InitializationEnum.ORTHOGONAL,
        bias_init: InitializationEnum = InitializationEnum.ZEROS,
    ):
        """
        Inits the LSTM block.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            num_layers (int): Number of stacked LSTM layers.
            batch_first (bool): Whether input tensors follow (batch, time, feat).
            forget_bias (float): Value assigned to forget gate bias.
            weights_init (InitializationEnum): Method used for input weights.
            recurrent_init (InitializationEnum): Method used for recurrent weights.
            bias_init (InitializationEnum): Method used for bias initialization.
        """
        super(LSTMBlock, self).__init__()
        self.hidden_dim = hidden_dim
        # Configure the LSTM layer, and initialize weights
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first)
        initializer: InitStrategy = LSTMInitStrategy(
            forget_bias, weights_init, recurrent_init, bias_init
        )
        initializer.initialize(self.lstm)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs an LSTM step sequence and returns updated memory.

        Args:
            x (torch.Tensor): Input sequence tensor.
            memory (torch.Tensor): Concatenated hidden and cell state.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                LSTM outputs and concatenated updated memory.
        """
        memory = memory.to(device=x.device, dtype=x.dtype)
        h0, c0 = torch.split(memory, self.hidden_dim, dim=-1)
        lstm_out, hidden_out = self.lstm(x, (h0, c0))
        memory_out = torch.cat(hidden_out, dim=-1)
        return lstm_out, memory_out

    def init_memory(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Creates an initial LSTM memory tensor.

        Args:
            batch_size (int): Current batch size.
            device (torch.device): Target tensor device.
            dtype (torch.dtype): Target tensor dtype.

        Returns:
            torch.Tensor: Concatenated initial memory of shape
                (num_layers, batch_size, 2 * hidden_dim).
        """
        num_layers = self.lstm.num_layers
        h0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        return torch.cat((h0, c0), dim=-1)


class CfCBlock(nn.Module):
    """
    Recurrent block wrapping a CfC layer and temporal state handling.
    """

    def __init__(
        self,
        input_dim: int,
        cfc_units: int,
        cfc_mode: str = "default",
        batch_first: bool = True,
        weights_init: InitializationEnum = InitializationEnum.XAVIER_UNIFORM,
        bias_init: InitializationEnum = InitializationEnum.ZEROS,
    ):
        """
        Inits the CfC block.

        Args:
            input_dim (int): Input feature dimension.
            cfc_units (int): CfC hidden/state dimension.
            cfc_mode (str): CfC mode ("default", "pure", or "no_gate").
            batch_first (bool): Whether inputs follow (batch, time, feat).
            weights_init (InitializationEnum): Method used for weight initialization.
            bias_init (InitializationEnum): Method used for bias initialization.
        """
        super(CfCBlock, self).__init__()
        self.cfc_units = cfc_units
        # Configure the CfC layer, and initialize weights
        self.cfc = CfC(
            input_dim,
            cfc_units,
            return_sequences=True,
            batch_first=batch_first,
            mode=cfc_mode,
        )
        initializer: InitStrategy = CfCInitStrategy(weights_init, bias_init)
        initializer.initialize(self.cfc)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a CfC sequence and returns updated memory.

        Args:
            x (torch.Tensor): Input tensor of shape (B, F) or (B, T, F).
            memory (torch.Tensor): CfC state. Supported shapes:
                (B, H) or ML-Agents style (1, B, H).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                CfC outputs and updated memory with shape (1, B, H).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Accept ML-Agents style memory [1, B, H] and normalize to [B, H].
        cfc_state_in = memory.squeeze(0) if memory.dim() == 3 else memory
        cfc_state_in = cfc_state_in.to(device=x.device, dtype=x.dtype)

        cfc_out, cfc_state = self.cfc(x, cfc_state_in)

        if cfc_out.dim() == 2:
            cfc_out = cfc_out.unsqueeze(1)

        if isinstance(cfc_state, tuple):
            cfc_state = cfc_state[0]
        if cfc_state.dim() == 1:
            cfc_state = cfc_state.unsqueeze(0)

        memory_out = cfc_state.unsqueeze(0).contiguous()
        return cfc_out, memory_out

    def init_memory(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Creates an initial CfC memory tensor.

        Args:
            batch_size (int): Current batch size.
            device (torch.device): Target tensor device.
            dtype (torch.dtype): Target tensor dtype.

        Returns:
            torch.Tensor: Initial memory of shape (1, batch_size, cfc_units).
        """
        return torch.zeros(1, batch_size, self.cfc_units, device=device, dtype=dtype)
