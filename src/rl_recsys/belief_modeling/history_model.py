import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


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

    def __init__(self, num_doc_features: int):
        super().__init__(num_doc_features=num_doc_features)

    def _init_history_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features).unsqueeze(0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the standardized avg of features of documents observed."""
        self.history_vec = torch.cat(
            (self.history_vec, observation.unsqueeze(0)), dim=0
        )
        history_retained = self.history_vec[-K:]
        std_hist_vec = torch.mean(history_retained, dim=0)

        # std_hist_vec = observation

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
        self.buffer = torch.zeros((1, 1, num_doc_features)).to(DEVICE)
        self.buffer_size = 100

    def _init_history_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(self.num_doc_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate buffer and input
        x = x.unsqueeze(0).unsqueeze(0).to(device=DEVICE)
        x = torch.cat([self.buffer, x], dim=1)

        # Update buffer
        self.buffer = x[:, -self.buffer_size :, :]

        # Forward pass through GRU and fully connected layers
        out, _ = self.gru(x)
        hist_vec = self.fc(out[:, -1, :]).squeeze(0)
        # print(hist_vec)
        return hist_vec
