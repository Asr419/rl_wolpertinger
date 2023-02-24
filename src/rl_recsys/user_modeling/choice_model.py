# implement user choice model
import abc
from typing import Any, Callable, Type, TypeVar

import numpy as np
import numpy.typing as npt


# maybe the scoring function can be passed as a parameter
class AbstractChoiceModel(metaclass=abc.ABCMeta):
    def __init__(self, **kwds: Any) -> None:
        pass

    @abc.abstractmethod
    def _score_document(
        self, user_state: npt.NDArray[np.float64], doc_repr: npt.NDArray[np.float64]
    ) -> float:
        pass

    def score_documents(
        self, user_state: npt.NDArray[np.float64], docs_repr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        scores = np.array([self._score_document(user_state, i) for i in docs_repr])
        return scores

    @abc.abstractmethod
    def choose_document(self, docs_repr: npt.NDArray[np.float64]) -> int:
        pass


class DeterministicChoicheModel(AbstractChoiceModel):
    """Select the doocument with the highest score with probability 1"""

    def _score_document(
        self, user_state: npt.NDArray[np.float64], doc_repr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.dot(user_state, doc_repr)

    def choose_document(self, scores: npt.NDArray[np.float64]) -> int:
        return int(np.argmax(scores))
