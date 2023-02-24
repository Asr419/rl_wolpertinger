import abc
from typing import Any

import numpy as np
import numpy.typing as npt


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


class DotProductResponseModel(AbstractResponseModel):
    def generate_response(
        self,
        estimated_user_state: npt.NDArray[np.float64],
        doc_repr: npt.NDArray[np.float64],
    ) -> float:
        """dot product response model"""
        r = np.dot(estimated_user_state, doc_repr)
        return r
