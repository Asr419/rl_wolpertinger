# implement user choice model
import abc
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


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

    def __init__(
        self, satisfaction_threshold: float = 0.2, no_selection_token: int = -1
    ) -> None:
        self.satisfaction_threshold = satisfaction_threshold
        self.no_selection_token = no_selection_token

    def choose_document(self) -> int:
        assert (
            self._scores is not None
        ), "Scores are not computed yet. call score_documents() first."
        # -1 indicates no document is selected
        selected_index = self.no_selection_token
        if torch.any(self._scores >= self.satisfaction_threshold):
            all_probs = torch.softmax(self._scores, dim=0)
            # select the item according to the probability distribution all_probs

            selected_index = int(torch.multinomial(all_probs, num_samples=1).item())

        return selected_index

    @abc.abstractmethod
    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        pass

    def score_documents(self, user_state: torch.Tensor, docs_repr: torch.Tensor):
        self._scores = self._score_documents(user_state, docs_repr)
        # normalize logits sum to 1
        # Use softmax scores instead of exponential scores to avoid overflow.


class DotProductChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model with dot product as the scoring function."""

    def __init__(self, satisfaction_threshold: float = 0.0) -> None:
        super().__init__(satisfaction_threshold)

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([torch.dot(user_state, doc_repr) for doc_repr in docs_repr])


class CosineSimilarityChoiceModel(NormalizableChoiceModel):
    """A multinomial logit choice model with cosine similarity as the scoring function."""

    def __init__(self, satisfaction_threshold: float = 0.0) -> None:
        super().__init__(satisfaction_threshold)

    def _score_documents(
        self, user_state: torch.Tensor, docs_repr: torch.Tensor
    ) -> torch.Tensor:
        # Calculate cosine similarity between user_state and each document representation
        scores = F.cosine_similarity(user_state, docs_repr)
        scores = (
            scores + 1
        ) / 2  # normalize cosine values to 0 and 1 for convenience of training
        # print(scores.max())
        return scores
