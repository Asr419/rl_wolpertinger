# implement user choice model
import abc
from typing import Any, Callable, Type, TypeVar

import numpy as np
import numpy.typing as npt


def softmax(vector: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Computes the softmax of a vector."""
    normalized_vector = np.array(vector) - np.max(vector)  # For numerical stability
    return np.exp(normalized_vector) / np.sum(np.exp(normalized_vector))


# maybe the scoring function can be passed as a parameter
class AbstractChoiceModel(metaclass=abc.ABCMeta):
    _scores = None

    @property
    def scores(self):
        return self._scores

    @abc.abstractmethod
    def _score_documents(
        self, user_state: npt.NDArray[np.float_], doc_repr: npt.NDArray[np.float_]
    ) -> float:
        pass

    @abc.abstractmethod
    def score_documents(
        self, user_state: npt.NDArray[np.float_], docs_repr: npt.NDArray[np.float_]
    ) -> None:
        pass

    @abc.abstractmethod
    def choose_document(self) -> int:
        # return the index of the chosen document in the slate
        pass


class NormalizableChoiceModel(AbstractChoiceModel):
    """A normalizable choice model."""

    def choose_document(self) -> int:
        assert (
            self._scores is not None
        ), "Scores are not computed yet. call score_documents() first."
        all_scores = self._scores
        all_probs = all_scores / np.sum(all_scores)
        selected_index = np.random.choice(len(all_probs), p=all_probs)
        return selected_index


class MultinomialLogitChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model.

    Samples item x in scores according to p(x) = exp(x) / Sum_{y in scores} exp(y)
    """

    @abc.abstractmethod
    def _score_documents(
        self, user_state: npt.NDArray[np.float_], docs_repr: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        pass

    def score_documents(
        self, user_state: npt.NDArray[np.float_], docs_repr: npt.NDArray[np.float_]
    ):
        logits = self._score_documents(user_state, docs_repr)
        # Use softmax scores instead of exponential scores to avoid overflow.
        self._scores = softmax(logits)


class DotProductChoiceModel(MultinomialLogitChoiceModel):
    """A multinomial logit choice model with dot product as the scoring function."""

    def _score_documents(
        self, user_state: npt.NDArray[np.float_], docs_repr: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        return np.array([np.dot(user_state, doc_repr) for doc_repr in docs_repr])
