import torch
import pytest

from config.settings import MemorySettings, NetworkSettings
from networks.init_strategy import (
    InitStrategy,
    InitializationEnum,
    LinearInitStrategy,
    LSTMInitStrategy,
)
from networks.layers import CfCBlock, LinearBlock, LSTMBlock
from networks.networkbody import NetworkBody


def test_linear_block_forward_shape() -> None:
    block = LinearBlock(input_dim=4, output_dim=16)
    x = torch.randn(5, 4)
    y = block(x)
    assert y.shape == (5, 16)


def test_linear_init_strategy_sets_bias_to_ones() -> None:
    layer = torch.nn.Linear(4, 8, bias=True)
    strategy = LinearInitStrategy(
        weights_init=InitializationEnum.XAVIER_UNIFORM,
        weights_gain=1.0,
        bias_init=InitializationEnum.ONES,
    )
    strategy.initialize(layer)
    assert torch.allclose(layer.bias, torch.ones_like(layer.bias))


def test_init_strategy_base_raises_not_implemented() -> None:
    base_strategy = InitStrategy()
    with pytest.raises(NotImplementedError):
        base_strategy.initialize(torch.nn.Linear(2, 2))


def test_lstm_block_init_memory_shape() -> None:
    block = LSTMBlock(input_dim=8, hidden_dim=16, num_layers=2)
    memory = block.init_memory(
        batch_size=3, device=torch.device("cpu"), dtype=torch.float32
    )
    assert memory.shape == (2, 3, 32)


def test_lstm_block_forward_shapes() -> None:
    block = LSTMBlock(input_dim=8, hidden_dim=16, num_layers=2)
    x = torch.randn(3, 4, 8)
    memory = block.init_memory(batch_size=3, device=x.device, dtype=x.dtype)
    out, memory_out = block(x, memory)
    assert out.shape == (3, 4, 16)
    assert memory_out.shape == (2, 3, 32)


def test_lstm_init_strategy_applies_forget_bias() -> None:
    forget_bias = 2.5
    layer = torch.nn.LSTM(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
    strategy = LSTMInitStrategy(
        forget_bias=forget_bias,
        weights_init=InitializationEnum.XAVIER_UNIFORM,
        recurrent_init=InitializationEnum.ORTHOGONAL,
        bias_init=InitializationEnum.ZEROS,
    )
    strategy.initialize(layer)

    bias_ih = layer.bias_ih_l0
    gate = bias_ih.shape[0] // 4
    forget_gate_slice = bias_ih[gate : 2 * gate]
    assert torch.allclose(
        forget_gate_slice, torch.full_like(forget_gate_slice, forget_bias)
    )


def test_cfc_block_init_memory_shape() -> None:
    block = CfCBlock(input_dim=8, cfc_units=12)
    memory = block.init_memory(
        batch_size=3, device=torch.device("cpu"), dtype=torch.float32
    )
    assert memory.shape == (1, 3, 12)


def test_cfc_block_forward_shapes() -> None:
    block = CfCBlock(input_dim=8, cfc_units=12)
    x = torch.randn(3, 5, 8)
    memory = block.init_memory(batch_size=3, device=x.device, dtype=x.dtype)
    out, memory_out = block(x, memory)
    assert out.shape == (3, 5, 12)
    assert memory_out.shape == (1, 3, 12)


def test_network_body_no_memory_returns_none_memory() -> None:
    settings = NetworkSettings(
        num_observations=4,
        num_layers=2,
        hidden_dim=16,
        weights_gain=1.41,
        memory=None,
    )
    model = NetworkBody(settings)
    x = torch.randn(3, 4)
    out, memory_out = model(x)
    assert out.shape == (3, 16)
    assert memory_out is None


def test_network_body_lstm_auto_init_memory() -> None:
    settings = NetworkSettings(
        num_observations=4,
        num_layers=2,
        hidden_dim=16,
        weights_gain=1.41,
        memory=MemorySettings(module_type="lstm", memory_dim=10),
    )
    model = NetworkBody(settings)
    x = torch.randn(3, 4)
    out, memory_out = model(x)
    assert out.shape == (3, 1, 10)
    assert memory_out is not None
    assert memory_out.shape == (1, 3, 20)


def test_network_body_cfc_auto_init_memory() -> None:
    settings = NetworkSettings(
        num_observations=4,
        num_layers=2,
        hidden_dim=16,
        weights_gain=1.41,
        memory=MemorySettings(module_type="cfc", memory_dim=9),
    )
    model = NetworkBody(settings)
    x = torch.randn(3, 4)
    out, memory_out = model(x)
    assert out.shape == (3, 1, 9)
    assert memory_out is not None
    assert memory_out.shape == (1, 3, 9)


def test_network_body_invalid_memory_type_raises() -> None:
    settings = NetworkSettings(
        num_observations=4,
        num_layers=2,
        hidden_dim=16,
        weights_gain=1.41,
        memory=MemorySettings(module_type="invalid", memory_dim=9),
    )
    with pytest.raises(ValueError):
        NetworkBody(settings)
