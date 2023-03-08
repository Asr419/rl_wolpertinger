import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractBeliefModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_doc_features: int) -> None:
        super().__init__()
        self.state = self._init_prev_state()
        self.num_doc_features = num_doc_features

    @abc.abstractmethod
    def _init_prev_state(self) -> torch.Tensor:
        # initailize the initial estimated state
        pass

    @abc.abstractmethod
    def forward(self, history_vec: torch.Tensor) -> torch.Tensor:
        """Outputs the estimated state given the history vector and the previous state

        Args:
            history_vec (torch.Tensor): vector representing history information

        Returns:
            torch.Tensor: estimated state
        """

        return self.state


class ConcatBeliefModel(AbstractBeliefModel):
    def __init__(self, num_doc_features: int) -> None:
        super().__init__(num_doc_features=num_doc_features)

    def _init_prev_vector(self) -> torch.Tensor:
        # initialize the history vector with zeros
        return torch.zeros(14)

    def forward(self, history_vec: torch.Tensor) -> torch.Tensor:
        """Outputs the estimated state given the history vector and the previous state

        Args:
            history_vec (torch.Tensor): vector representing history information

        Returns:
            torch.Tensor: estimated state
        """
        self.state = torch.cat((self.state, history_vec), dim=0)
        return self.state


class NNBeliefModel(AbstractBeliefModel):
    def __init__(self, num_doc_features: int) -> None:
        super(NNBeliefModel, self).__init__(num_doc_features=num_doc_features)
        input_size = 2 * num_doc_features
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_doc_features)

    def forward(self, history_vec: torch.Tensor) -> torch.Tensor:
        """Outputs the estimated state given the history vector and the previous state

        Args:
            history_vec (torch.Tensor): vector representing history information

        Returns:
            torch.Tensor: estimated state
        """
        x = torch.cat((self.state, history_vec), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        self.state = self.fc3(x)
        return self.state
