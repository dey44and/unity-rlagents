from enum import Enum 
from typing import Callable
import torch
import torch.nn as nn


class InitializationEnum(Enum):
    """
    Enumerates supported parameter initialization methods.
    """
    ZEROS = "zeros"
    ONES = "ones"
    XAVIER_NORMAL = "xavier_normal"
    XAVIER_UNIFORM = "xavier_uniform"
    ORTHOGONAL = "orthogonal"


WEIGHT_INIT_METHODS: dict[InitializationEnum, Callable[[torch.Tensor], torch.Tensor]] = {
    InitializationEnum.XAVIER_NORMAL:  nn.init.xavier_normal_,
    InitializationEnum.XAVIER_UNIFORM: nn.init.xavier_uniform_,
    InitializationEnum.ORTHOGONAL: nn.init.orthogonal_
}


BIAS_INIT_METHODS: dict[InitializationEnum, Callable[[torch.Tensor], torch.Tensor]] = {
    InitializationEnum.ZEROS: nn.init.zeros_,
    InitializationEnum.ONES: nn.init.ones_
}


class InitStrategy:
    """
    Base strategy interface for initializing neural network modules.
    """
    def initialize(self, module: nn.Module):
        """
        Initializes the provided module in-place.

        Args:
            module (nn.Module): Module to initialize.
        """
        raise NotImplementedError("Method should be implemented in subclasses")


class LinearInitStrategy(InitStrategy):
    """
    Initialization strategy for linear layers.
    """
    def __init__(
        self,
        weights_init: InitializationEnum = InitializationEnum.ORTHOGONAL, 
        weights_gain: float = 1.0,
        bias_init: InitializationEnum = InitializationEnum.ZEROS
    ):
        """
        Inits the linear initialization strategy.

        Args:
            weights_init (InitializationEnum): Method used for weight initialization.
            weights_gain (float): Gain multiplier applied after weight initialization.
            bias_init (InitializationEnum): Method used for bias initialization.
        """
        super().__init__()
        self.weights_init = weights_init
        self.weights_gain = weights_gain
        self.bias_init = bias_init

    def initialize(self, module: nn.Module):
        """
        Applies configured initialization to a linear module.

        Args:
            module (nn.Module): Target linear module.
        """
        with torch.no_grad():
            WEIGHT_INIT_METHODS[self.weights_init](module.weight)
            module.weight.mul_(self.weights_gain)
            BIAS_INIT_METHODS[self.bias_init](module.bias)


class LSTMInitStrategy(InitStrategy):
    """
    Initialization strategy for LSTM layers with gate-aware setup.
    """
    def __init__(
        self,
        forget_bias: float = 1.0,
        weights_init: InitializationEnum = InitializationEnum.XAVIER_UNIFORM,
        recurrent_init: InitializationEnum = InitializationEnum.ORTHOGONAL,
        bias_init: InitializationEnum = InitializationEnum.ZEROS
    ):
        """
        Inits the LSTM initialization strategy.

        Args:
            forget_bias (float): Value assigned to forget gate bias.
            weights_init (InitializationEnum): Method used for input weights.
            recurrent_init (InitializationEnum): Method used for recurrent weights.
            bias_init (InitializationEnum): Method used for all non-forget biases.
        """
        super().__init__()
        self.forget_bias = forget_bias
        self.weights_init = weights_init
        self.recurrent_init = recurrent_init
        self.bias_init = bias_init

    def initialize(self, module: nn.Module):
        """
        Applies configured initialization to an LSTM module.

        Args:
            module (nn.Module): Target LSTM module.
        """
        with torch.no_grad():
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    gate = param.shape[0] // 4
                    for i in range(4):
                        block = param[i * gate : (i + 1) * gate]
                        WEIGHT_INIT_METHODS[self.weights_init](block)

                elif "weight_hh" in name:
                    gate = param.shape[0] // 4
                    for i in range(4):
                        block = param[i * gate : (i + 1) * gate]
                        WEIGHT_INIT_METHODS[self.recurrent_init](block)

                elif "bias_ih" in name:
                    BIAS_INIT_METHODS[self.bias_init](param)
                    gate = param.shape[0] // 4
                    param[gate : 2 * gate].fill_(self.forget_bias)  # forget gate

                elif "bias_hh" in name:
                    BIAS_INIT_METHODS[self.bias_init](param)
