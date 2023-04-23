import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


class AbstractResponseModel(metaclass=abc.ABCMeta):
    def __init__(self, null_response: float = -1.0) -> None:
        self.null_response = null_response

    @abc.abstractmethod
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        """
        Generate the user response (reward) to a slate,
        is a function of the user state and the chosen document in the slate.

        Args:
            estimated_user_state (np.array): estimated user state
            doc_repr (np.array): document representation

        Returns:
            float: user response
        """
        pass

    def generate_null_response(self) -> torch.Tensor:
        return torch.tensor(self.null_response)


class AmplifiedResponseModel(AbstractResponseModel):
    def __init__(self, amp_factor: int = 10, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.amp_factor = amp_factor

    @abc.abstractmethod
    def _generate_response(
        self,
        estimated_user_state: npt.NDArray[np.float64],
        doc_repr: npt.NDArray[np.float64],
    ) -> float:
        """
        Generate the user response (reward) to a slate,
        is a function of the user state and the chosen document in the slate.

        Args:
            estimated_user_state (np.array): estimated user state
            doc_repr (np.array): document representation

        Returns:
            float: user response
        """
        pass

    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        return self._generate_response(estimated_user_state, doc_repr) * self.amp_factor

    def generate_null_response(self) -> float:
        return super().generate_null_response() * self.amp_factor
    
    def generate_topic_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        
        doc_item=doc_repr[:20]
        
        doc_length = doc_repr[20:21]
        doc_quality = doc_repr[21:22]
        return self._generate_response(estimated_user_state, doc_item) * (1-self.amp_factor) + self.amp_factor*doc_quality


class CosineResponseModel(AmplifiedResponseModel):
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        satisfaction = torch.nn.functional.cosine_similarity(
            estimated_user_state, doc_repr, dim=0
        )
        return satisfaction


class DotProductResponseModel(AmplifiedResponseModel):
    def _generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        satisfaction = torch.dot(estimated_user_state, doc_repr)
        return satisfaction
