import abc
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_recsys.belief_modeling.history_model import AbstractHistoryModel

hist_model_type = TypeVar("hist_model_type", bound=AbstractHistoryModel)


class AbstractBeliefModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_doc_features: int, hist_model: hist_model_type) -> None:
        super().__init__()
        self.state = None
        self.num_doc_features = num_doc_features
        self.hist = hist_model

    @abc.abstractmethod
    def _init_prev_state(self, initial_state: torch.Tensor) -> None:
        # initailize the initial estimated state
        self.state = initial_state

    @abc.abstractmethod
    def forward(self, history_vec: torch.Tensor) -> torch.Tensor:
        """Outputs the estimated state given the history vector and the previous state

        Args:
            history_vec (torch.Tensor): vector representing history information

        Returns:
            torch.Tensor: estimated state
        """

        return self.state


class NNBeliefModel(AbstractBeliefModel):
    def __init__(
        self,
        num_doc_features: int,
        hist_model: hist_model_type,
        hidden_dims: list[int] = [14],
    ) -> None:
        super(NNBeliefModel, self).__init__(
            num_doc_features=num_doc_features, hist_model=hist_model
        )

        input_size = 2 * num_doc_features
        output_size = num_doc_features

        self.history_model = hist_model
        self.layers = nn.ModuleList()
        # create layers based on hidden_dims
        prev_dim = input_size
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        # add last layer
        self.layers.append(nn.Linear(prev_dim, output_size))

    def forward(self, selcted_doc_features: torch.Tensor) -> torch.Tensor:
        """Outputs the estimated state given the history vector and the previous state"""
        if self.state is None:
            raise ValueError("Initial state not set")
        
        history_vec = self.history_model(selcted_doc_features)

        # concatenate history and state
        x = torch.cat((self.state, history_vec), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x)
        # set state
        self.state = x
        return self.state
