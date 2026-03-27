import torch

from networks.modules.networkbody import NetworkBody
from config.settings import NetworkSettings, CfCMemorySettings, LSTMMemorySettings

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    memory = CfCMemorySettings(memory_dim=32, mode="default")
    netsettings = NetworkSettings(
        num_observations=4,
        num_layers=2,
        hidden_dim=32,
        weights_gain=1.41,
        memory="cfc",
        memory_settings=memory,
    )

    base: NetworkBody = NetworkBody(netsettings=netsettings).to(device)

    x = torch.rand(1, 4).to(device)
    y, memory = base(x)

    print(y)
