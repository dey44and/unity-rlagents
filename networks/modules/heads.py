import torch
import torch.nn as nn
from typing import List


class SimpleCategoricalHead(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiCategoricalHead(nn.Module):
    def __init__(self, simple_heads: List[SimpleCategoricalHead]):
        super().__init__()
        self.simple_heads = simple_heads

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [simple_head(x) for simple_head in self.simple_heads]
