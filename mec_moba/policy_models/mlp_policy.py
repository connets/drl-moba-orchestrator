import torch, math, numpy as np
from torch import nn


class MLPNet(nn.Module):
    def __init__(self, state_shape, action_shape, layer_dim=None, num_layers=2):
        super().__init__()
        if layer_dim is None:
            layer_dim = pow(2, math.floor(math.log2(max(np.prod(state_shape), np.prod(action_shape)))))

        layers = [nn.Linear(np.prod(state_shape), layer_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers):
            layers.append(nn.Linear(layer_dim, layer_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layer_dim, np.prod(action_shape)))

        self.model = nn.Sequential(*layers)

        # self.model = nn.Sequential(
        #     nn.Linear(np.prod(state_shape), layer_dim), nn.ReLU(inplace=True),
        #     nn.Linear(layer_dim, layer_dim), nn.ReLU(inplace=True),
        #     nn.Linear(layer_dim, layer_dim), nn.ReLU(inplace=True),
        #     nn.Linear(layer_dim, np.prod(action_shape)),
        # )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
