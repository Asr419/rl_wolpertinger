import numpy as np
from rl_recsys.agent_modeling.dqn_agent import DQNnet
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.utils import load_spotify_data
from rl_recsys.agent_modeling.slate_generator import TopK_slate
import torch


def test_top_K():
    data_df = load_spotify_data()

    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")
    dummy_belief_state = torch.rand(14)

    doc_ids = np.array([5, 7, 9, 21, 15, 19, 11, 56, 47, 95])
    topKAgent = TopK_slate(dummy_belief_state, candidate_docs=doc_ids)
    topK = topKAgent.topk_slate(dummy_belief_state, doc_ids)
    a = 5
