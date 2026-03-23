from .init_strategy import *

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
        super().__init__()
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
        super().__init__()
        self.hidden_dim = hidden_dim
        # Configure the LSTM layer, and initialize weights
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first)
        initializer: InitStrategy = LSTMInitStrategy(
            forget_bias, weights_init, recurrent_init, bias_init
        )
        initializer.initialize(self.lstm)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        """
        Runs an LSTM step sequence and returns updated memory.

        Args:
            x (torch.Tensor): Input sequence tensor.
            memory (torch.Tensor): Concatenated hidden and cell state.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                LSTM outputs and concatenated updated memory.
        """
        h0, c0 = torch.split(memory, self.hidden_dim, dim=-1)
        lstm_out, hidden_out = self.lstm(x, (h0, c0))
        memory_out = torch.cat(hidden_out, dim=-1)
        return lstm_out, memory_out


class CfCBlock(nn.Module):
    """
    Placeholder block for a future Closed-form Continuous-time module.
    """

    def __init__(self):
        """
        Inits the CfC block placeholder.
        """
        pass

    def forward(self, x: torch.Tensor):
        """
        Placeholder forward pass.

        Args:
            x (torch.Tensor): Input tensor.
        """
        pass
