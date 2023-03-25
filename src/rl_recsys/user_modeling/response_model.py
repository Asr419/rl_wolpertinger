import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


class AbstractResponseModel(metaclass=abc.ABCMeta):
    def __init__(self, **kwds: Any) -> None:
        pass

    @abc.abstractmethod
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate the user response (reward) to a slate,
        is a function of the user state and the chosen document in the slate.

        Args:
            estimated_user_state (torch.Tensor): estimated user state
            doc_repr (torch.Tensor): document representation

        Returns:
            torch.Tensor: user response
        """
        pass


class DotProductResponseModel(AbstractResponseModel):
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> torch.Tensor:
        """dot product response model"""
        return torch.dot(estimated_user_state, doc_repr)
