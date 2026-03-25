from .layers import CfCBlock, LSTMBlock
from typing import Union
from config.settings import LSTMMemorySettings, CfCMemorySettings
import torch.nn as nn

_memory_builders = {
    "lstm": lambda cfg, in_dim: LSTMBlock(
        input_dim=in_dim, hidden_dim=cfg.memory_dim, num_layers=cfg.num_layers
    ),
    "cfc": lambda cfg, in_dim: CfCBlock(
        input_dim=in_dim, cfc_units=cfg.memory_dim, cfc_mode=cfg.mode
    ),
}


class MemModuleCreator:
    """
    Factory utility that creates recurrent memory modules from validated settings.
    """

    @staticmethod
    def create(
        memory_type: str,
        input_dim: int,
        config: Union[LSTMMemorySettings, CfCMemorySettings],
    ) -> nn.Module:
        """
        Builds the configured recurrent module.

        Args:
            memory_type (str): Memory module identifier ("lstm" or "cfc").
            input_dim (int): Feature dimension consumed by the recurrent block.
            config (Union[LSTMMemorySettings, CfCMemorySettings]): Typed memory config.

        Returns:
            nn.Module: Instantiated recurrent module.

        Raises:
            ValueError: If the provided memory type is not supported.
        """
        if memory_type not in _memory_builders.keys():
            raise ValueError(
                f"Expected {list(_memory_builders.keys())}, got '{memory_type}'"
            )
        return _memory_builders[memory_type](config, input_dim)
