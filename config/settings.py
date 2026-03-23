from typing import NamedTuple, Optional


class MemorySettings(NamedTuple):
    """
    Immutable configuration container for the memory module.
    """

    module_type: str
    memory_dim: int


class NetworkSettings(NamedTuple):
    """
    Immutable configuration container for the network body.
    """

    num_observations: int
    num_layers: int
    hidden_dim: int
    weights_gain: float
    memory: Optional[MemorySettings]
