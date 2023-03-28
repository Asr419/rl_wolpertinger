import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

LAMBDA = 10


class AbstractResponseModel(metaclass=abc.ABCMeta):
    def __init__(self, **kwds: Any) -> None:
        pass

    @abc.abstractmethod
    def generate_response(
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

        # class DotProductResponseModel(AbstractResponseModel):
        #     def generate_response(
        #         self,
        #         estimated_user_state: npt.NDArray[np.float64],
        #         doc_repr: npt.NDArray[np.float64],
        #     ) -> float:
        #         """dot product response model"""
        #         r = np.dot(estimated_user_state, doc_repr)
        #         return r

        """cosine response model"""


class DotProductResponseModel(AbstractResponseModel):
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        """dot product response model"""
        r = torch.dot(estimated_user_state, doc_repr)
        return r


# class CosineResponseModel(AbstractResponseModel):
#     def generate_response(
#         self,
#         estimated_user_state: npt.NDArray[np.float64],
#         doc_repr: npt.NDArray[np.float64],
#     ) -> float:
#         r = LAMBDA * (
#             np.dot(estimated_user_state, doc_repr)
#             / (np.linalg.norm(estimated_user_state) * np.linalg.norm(doc_repr))
#         )

#         # cos_sim = torch.nn.functional.cosine_similarity(estimated tensor2, dim=0)
#         return r


class CosineResponseModel(AbstractResponseModel):
    def generate_response(
        self,
        estimated_user_state: torch.Tensor,
        doc_repr: torch.Tensor,
    ) -> float:
        cos_sim = torch.nn.functional.cosine_similarity(
            estimated_user_state, doc_repr, dim=0
        )
        r = LAMBDA * cos_sim
        return r

    def Amplifier(self):
        return LAMBDA
