import abc
from typing import Tuple

import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class WolpertingerActor(nn.Module):
    def __init__(self, nn_dim: list[int], k: int, input_dim: int = 14):
        super(WolpertingerActor, self).__init__()
        self.k = k

        layers = []
        for i, dim in enumerate(nn_dim):
            if i == 0:
                layers.append(nn.Linear(input_dim, dim))
            elif i == len(nn_dim) - 1:
                layers.append(nn.Linear(dim, 14))
            else:
                layers.append(nn.Linear(dim, dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # output protoaction
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x

    def k_nearest(
        self,
        input_state: torch.Tensor,
        candidate_docs: torch.Tensor,
    ) -> None:
        proto_action = self(input_state)
        distances = torch.linalg.norm(candidate_docs - proto_action, axis=1)
        # Sort distances and get indices of k smallest distances
        indices = torch.argsort(distances, dim=0)[: self.k]
        # Select k closest tensors from tensor list
        candidates_subset = candidate_docs[indices]
        return candidates_subset
