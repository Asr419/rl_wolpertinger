import abc
from typing import Tuple

import numpy as np
import torch

from rl_recsys.agent_modeling.dqn_agent import DQNnet
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.utils import load_spotify_data


class AbstractSlateGenerator(metaclass=abc.ABCMeta):
    def __init__(self, slate_size: int = 10) -> None:
        self.slate_size = slate_size

    @abc.abstractmethod
    def __call__(
        self,
        docs_id: torch.Tensor,
        docs_scores: torch.Tensor,
        docs_qvalues: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a state and a set of candidate documents, create a slate of documents"""
        pass


class TopKSlateGenerator(AbstractSlateGenerator):
    def __call__(
        self, docs_scores: torch.Tensor, docs_qvalues: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_scores, topk_ids = torch.topk(
            docs_scores * docs_qvalues, k=self.slate_size
        )
        return topk_scores, topk_ids
