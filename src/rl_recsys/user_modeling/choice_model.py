# implement user choice model
import abc

import numpy as np
import numpy.typing as npt


class AbstractChoiceModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _score_document(self, doc_repr: npt.NDArray[np.float64]) -> float:
        pass

    def score_documents(
        self, docs_repr: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        scores = np.array([self._score_document(i) for i in docs_repr])
        return scores
