import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractHistoryModel(nn.Module, metaclass=abc.ABCMeta):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int, **kwargs):
        super().__init__()
        self.num_doc_features = num_doc_features

    @abc.abstractmethod
    def forward(self, obs_buff: torch.Tensor) -> torch.Tensor:
        """Given the last observation, return a vector representing history information."""
        # todo: maybe here we can have a standradization step
        pass


class LastObservationModel(AbstractHistoryModel):
    """Modeling session history information."""

    def forward(self, obs_buff: torch.Tensor) -> torch.Tensor:
        """Return last observation."""
        hist_vec = None
        if len(obs_buff.shape) == 3:
            hist_vec = obs_buff[:, -1, :]
        elif len(obs_buff.shape) == 2:
            hist_vec = obs_buff[-1, :]
        else:
            raise ValueError("obs_buff shape is not correct")
        return hist_vec  # type: ignore


class AvgHistoryModel(AbstractHistoryModel):
    """Modeling session history information."""

    def __init__(self, num_doc_features: int, memory_length: int = 10):
        super().__init__(num_doc_features=num_doc_features)
        self.memory_length = memory_length

    def forward(self, obs_buff: torch.Tensor) -> torch.Tensor:
        """Return the standardized avg of features of documents observed."""
        hist_vec = None
        if len(obs_buff.shape) == 3:
            hist_vec = obs_buff[:, -self.memory_length :, :].mean(dim=1)
        elif len(obs_buff.shape) == 2:
            hist_vec = obs_buff[-self.memory_length :, :].mean(dim=0)
        else:
            raise ValueError("obs_buff shape is not correct")

        return hist_vec  # type: ignore


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out
