import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractBeliefModel(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.prev_state = self._init_prev_state()

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
        pass


class ConcatBeliefModel(AbstractBeliefModel):
    pass
