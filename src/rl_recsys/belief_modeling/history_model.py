import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractHistoryModel(nn.Module, metaclass=abc.ABCMeta):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int):
        super().__init__()
        self.num_doc_features = num_doc_features
        self.history_vec = self._init_history_vectror()

    @abc.abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Given the last observation, return a vector representing history information."""
        # todo: maybe here we can have a standradization step
        pass

    @abc.abstractmethod
    def _init_history_vectror(self) -> torch.Tensor:
        """Generate the first history vector."""
        pass


class AvgHistoryModel(AbstractHistoryModel):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int):
        super().__init__(num_doc_features=num_doc_features)

    def _init_history_vectror(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the standardized avg of features of documents observed."""
        # TODO: wronggggg
        hist_vec = (self.history_vec + observation) / 2
        std_hist_vec = (hist_vec - torch.mean(hist_vec)) / torch.std(hist_vec)
        return std_hist_vec
