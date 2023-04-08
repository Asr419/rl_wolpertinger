import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractHistoryModel(nn.Module, metaclass=abc.ABCMeta):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int):
        super().__init__()
        self.num_doc_features = num_doc_features

        # register buffer for history vector
        self.register_buffer("history_vec", self._init_history_vector())
        # self.history_vec = self._init_history_vector()

    @abc.abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Given the last observation, return a vector representing history information."""
        # todo: maybe here we can have a standradization step
        pass

    @abc.abstractmethod
    def _init_history_vector(self) -> torch.Tensor:
        """Generate the first history vector."""
        pass


class AvgHistoryModel(AbstractHistoryModel):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int, memory_length: int = 10):
        super().__init__(num_doc_features=num_doc_features)
        self.memory_length = memory_length

    def _init_history_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features).unsqueeze(0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the standardized avg of features of documents observed."""
        self.history_vec = torch.cat(
            (self.history_vec, observation.unsqueeze(0)), dim=0
        )
        history_retained = self.history_vec[-self.memory_length :]
        std_hist_vec = torch.mean(history_retained, dim=0)

        # std_hist_vec = observation

        return std_hist_vec


class GRUModel(AbstractHistoryModel):
    def __init__(
        self,
        num_doc_features: int,
        hidden_size: int = 14,
        num_layers: int = 1,
    ):
        super().__init__(num_doc_features=num_doc_features)

        self.gru = nn.GRU(
            input_size=num_doc_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def _init_history_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out
