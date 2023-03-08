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
    def _init_history_vector(self) -> torch.Tensor:
        """Generate the first history vector."""
        pass


class AvgHistoryModel(AbstractHistoryModel):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int):
        super().__init__(num_doc_features=num_doc_features)

    def _init_history_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the standardized avg of features of documents observed."""
        # TODO: wronggggg
        hist_vec = (self.history_vec + observation) / 2
        std_hist_vec = (hist_vec - torch.mean(hist_vec)) / torch.std(hist_vec)
        return std_hist_vec


class GRUModel(AbstractHistoryModel):
    def __init__(
        self,
        num_doc_features: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
    ):
        super().__init__(num_doc_features=num_doc_features)
        self.gru = nn.GRU(
            input_size=num_doc_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize buffer
        self.buffer = torch.zeros((1, 1, num_doc_features))
        self.buffer_size = 100

    def _init_history_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate buffer and input
        x = torch.cat((self.buffer, x), dim=1)

        # Update buffer
        self.buffer = x[:, -self.buffer_size :, :]

        # Forward pass through GRU and fully connected layers
        out, _ = self.gru(x)
        hist_vec = self.fc(out[:, -1, :])

        return hist_vec
