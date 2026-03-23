from typing import NamedTuple


class NetworkSettings(NamedTuple):
    """
    Immutable configuration container for the network body.
    """

    num_observations: int
    num_layers: int
    hidden_dim: int
    weights_gain: float
    use_memory: bool
