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


class TopK_slate:
    def __init__(self, state, candidate_docs, input_size=28):
        self.state = state
        self.candidate_docs = candidate_docs
        self.input_size = input_size

    def topk_slate(self, state, candidate_docs):
        data_df = load_spotify_data()
        doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")
        output_size = 1
        slate_scores = []
        self.q_net = DQNnet(self.input_size, output_size)
        self.targetq_net = DQNnet(self.input_size, output_size)
        for i in candidate_docs:
            doc_feature = doc_catalogue.get_doc_features(doc_id=i)
            feature_tensor = torch.Tensor(doc_feature)
            x = torch.cat((state, feature_tensor), dim=0)
            with torch.no_grad():
                predicted = self.q_net.forward(x)
            predicted = predicted.numpy()
            predicted = np.dot(state, doc_feature) * predicted
            slate_scores.append(predicted[0])
        slate_scores_tensor = torch.stack([torch.tensor(arr) for arr in slate_scores])
        # slate_scores_tensor = torch.Tensor(slate_scores)
        values, idx = torch.topk(slate_scores_tensor, k=10, axis=-1)
        idx = idx.numpy()
        slate_topk = candidate_docs[idx]
        return slate_topk
