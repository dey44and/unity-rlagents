from typing import NamedTuple, Optional, Union


class LSTMMemorySettings(NamedTuple):
    """
    Immutable configuration container for the lstm memory module.
    """

    memory_dim: int
    num_layers: int


class CfCMemorySettings(NamedTuple):
    """
    Immutable configuration container for the cfc memory module.
    """

    memory_dim: int
    mode: str


class NetworkSettings(NamedTuple):
    """
    Immutable configuration container for the network body.
    """

    num_observations: int
    num_layers: int
    hidden_dim: int
    weights_gain: float
    memory: str
    memory_settings: Optional[Union[LSTMMemorySettings, CfCMemorySettings]]
