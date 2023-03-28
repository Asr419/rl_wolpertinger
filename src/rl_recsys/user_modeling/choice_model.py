# implement user choice model
import abc
from typing import Any, Callable, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


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
        self, user_state: torch.Tensor, doc_repr: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
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
        if torch.any(all_scores >= 0.9):
            all_probs = torch.softmax(all_scores, dim=0)
            # select index according to the probability distribution with pytorch
            selected_index = torch.multinomial(all_probs, 1).item()
            return selected_index
        else:
            selected_index = -1
            return selected_index

    @abc.abstractmethod
    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        pass

    def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
        logits = self._score_documents(user_state, docs_repr)
        # Use softmax scores instead of exponential scores to avoid overflow.
        self._scores = logits


# class MultinomialLogitChoiceModel(NormalizableChoiceModel):
#     """A multinomial logit choice model.

#     Samples item x in scores according to p(x) = exp(x) / Sum_{y in scores} exp(y)
#     """

#     @abc.abstractmethod
#     def _score_documents(
#         self, user_state: torch.Tensor, docs_repr: torch.Tensor
#     ) -> torch.Tensor:
#         pass

#     def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
#         logits = self._score_documents(user_state, docs_repr)
#         # Use softmax scores instead of exponential scores to avoid overflow.
#         self._scores = torch.softmax(logits, dim=0)


class DotProductChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model with dot product as the scoring function."""

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([torch.dot(user_state, doc_repr) for doc_repr in docs_repr])


class CosineSimilarityChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model with cosine similarity as the scoring function."""

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        # Calculate cosine similarity between user_state and each document representation
        return F.cosine_similarity(user_state, docs_repr)
