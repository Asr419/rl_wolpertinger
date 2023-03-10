import abc

import numpy as np
import torch

from rl_recsys.agent_modeling.dqn_agent import DQNnet
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.utils import load_spotify_data


class AbstractSlateGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        candidate_docs: torch.Tensor,
    ) -> torch.Tensor:
        """Given a state and a set of candidate documents, create a slate of documents"""
        pass


class TopKSlateGenerator(AbstractSlateGenerator):
    pass


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


# def compute_probs(slate, scores_tf, score_no_click_tf):
#     """Computes the selection probability and returns selected index.
#     This assumes scores are normalizable, e.g., scores cannot be negative.
#     Args:
#       slate: a list of integers that represents the video slate.
#       scores_tf: a float tensor that stores the scores of all documents.
#       score_no_click_tf: a float tensor that represents the score for the action
#         of picking no document.
#     Returns:
#       A float tensor that represents the probabilities of selecting each document
#         in the slate.
#     """
#     all_scores = torch.concat(
#         [torch.gather(scores_tf, slate), torch.reshape(score_no_click_tf, (1, 1))],
#         axis=0,
#     )  # pyformat: disable
#     all_probs = all_scores / torch.sum(input_tensor=all_scores)
#     return all_probs[:-1]


# def score_documents_tf(
#     user_obs, doc_obs, no_click_mass=1.0, is_mnl=False, min_normalizer=-1.0
# ):
#     """Computes unnormalized scores given both user and document observations.
#     This implements both multinomial proportional model and multinormial logit
#       model given some parameters. We also assume scores are based on inner
#       products of user_obs and doc_obs.
#     Args:
#       user_obs: An instance of AbstractUserState.
#       doc_obs: A numpy array that represents the observation of all documents in
#         the candidate set.
#       no_click_mass: a float indicating the mass given to a no click option
#       is_mnl: whether to use a multinomial logit model instead of a multinomial
#         proportional model.
#       min_normalizer: A float (<= 0) used to offset the scores to be positive when
#         using multinomial proportional model.
#     Returns:
#       A float tensor that stores unnormalzied scores of documents and a float
#         tensor that represents the score for the action of picking no document.
#     """
#     user_obs = torch.reshape(user_obs, [1, -1])
#     scores = torch.sum(input_tensor=torch.multiply(user_obs, doc_obs), axis=1)
#     all_scores = torch.concat([scores, torch.constant([no_click_mass])], axis=0)
#     if is_mnl:
#         all_scores = torch.nn.functional.softmax(all_scores)
#     else:
#         all_scores = all_scores - min_normalizer
#     return all_scores[:-1], all_scores[-1]


# def score_documents(
#     user_obs, doc_obs, no_click_mass=1.0, is_mnl=False, min_normalizer=-1.0
# ):
#     """Computes unnormalized scores given both user and document observations.
#     Similar to score_documents_tf but works on NumPy objects.
#     Args:
#       user_obs: An instance of AbstractUserState.
#       doc_obs: A numpy array that represents the observation of all documents in
#         the candidate set.
#       no_click_mass: a float indicating the mass given to a no click option
#       is_mnl: whether to use a multinomial logit model instead of a multinomial
#         proportional model.
#       min_normalizer: A float (<= 0) used to offset the scores to be positive when
#         using multinomial proportional model.
#     Returns:
#       A float array that stores unnormalzied scores of documents and a float
#         number that represents the score for the action of picking no document.
#     """
#     scores = np.array([])
#     for doc in doc_obs:
#         scores = np.append(scores, np.dot(user_obs, doc))

#     all_scores = np.append(scores, no_click_mass)
#     if is_mnl:
#         all_scores = choice_model.softmax(all_scores)
#     else:
#         all_scores = all_scores - min_normalizer
#     assert not all_scores[
#         all_scores < 0.0
#     ], "Normalized scores have non-positive elements."
#     return all_scores[:-1], all_scores[-1]


# def select_slate_topk(slate_size, s_no_click, s, q):
#     """Selects the slate using the top-K algorithm.
#     This algorithm corresponds to the method "TS" in
#     Ie et al. https://arxiv.org/abs/1905.12767.
#     Args:
#       slate_size: int, the size of the recommendation slate.
#       s_no_click: float tensor, the score for not clicking any document.
#       s: [num_of_documents] tensor, the scores for clicking documents.
#       q: [num_of_documents] tensor, the predicted q values for documents.
#     Returns:
#       [slate_size] tensor, the selected slate.
#     """
#     del s_no_click  # Unused argument.
#     _, output_slate = torch.top_k(s * q, k=slate_size)
#     return output_slate


# def select_slate_greedy(slate_size, s_no_click, s, q):
#     """Selects the slate using the adaptive greedy algorithm.
#     This algorithm corresponds to the method "GS" in
#     Ie et al. https://arxiv.org/abs/1905.12767.
#     Args:
#       slate_size: int, the size of the recommendation slate.
#       s_no_click: float tensor, the score for not clicking any document.
#       s: [num_of_documents] tensor, the scores for clicking documents.
#       q: [num_of_documents] tensor, the predicted q values for documents.
#     Returns:
#       [slate_size] tensor, the selected slate.
#     """

#     def argmax(v, mask):
#         return torch.argmax(input=(v - torch.min(input_tensor=v) + 1) * mask, axis=0)

#     numerator = torch.constant(0.0)
#     denominator = torch.constant(0.0) + s_no_click
#     mask = torch.ones(torch.shape(input=q)[0])

#     def set_element(v, i, x):
#         mask = torch.nn.functional.one_hot(i, torch.shape(input=v)[0])
#         v_new = torch.ones_like(v) * x
#         return torch.where(torch.equal(mask, 1), v_new, v)

#     for _ in range(slate_size):
#         k = argmax((numerator + s * q) / (denominator + s), mask)
#         mask = set_element(mask, k, 0)
#         numerator = numerator + torch.gather(s * q, k)
#         denominator = denominator + torch.gather(s, k)

#     output_slate = torch.where(torch.equal(mask, 0))
#     return output_slate


# def select_slate_optimal(slate_size, s_no_click, s, q):
#     """Selects the slate using exhaustive search.
#     This algorithm corresponds to the method "OS" in
#     Ie et al. https://arxiv.org/abs/1905.12767.
#     Args:
#       slate_size: int, the size of the recommendation slate.
#       s_no_click: float tensor, the score for not clicking any document.
#       s: [num_of_documents] tensor, the scores for clicking documents.
#       q: [num_of_documents] tensor, the predicted q values for documents.
#     Returns:
#       [slate_size] tensor, the selected slate.
#     """

#     num_candidates = s.shape.as_list()[0]

#     # Obtain all possible slates given current docs in the candidate set.
#     mesh_args = [list(range(num_candidates))] * slate_size
#     slates = torch.stack(torch.meshgrid(*mesh_args), axis=-1)
#     slates = torch.reshape(slates, shape=(-1, slate_size))

#     # Filter slates that include duplicates to ensure each document is picked
#     # at most once.
#     unique_mask = tf.map_fn(
#         lambda x: torch.equal(tf.size(input=x), tf.size(input=tf.unique(x)[0])),
#         slates,
#         dtype=torch.tensor.bool,
#     )
#     slates = torch.masked_select(tensor=slates, mask=unique_mask)

#     slate_q_values = torch.gather(s * q, slates)
#     slate_scores = torch.gather(s, slates)
#     slate_normalizer = torch.reduce_sum(input_tensor=slate_scores, axis=1) + s_no_click

#     slate_q_values = slate_q_values / tf.expand_dims(slate_normalizer, 1)
#     slate_sum_q_values = torch.sum(input_tensor=slate_q_values, axis=1)
#     max_q_slate_index = torch.argmax(input=slate_sum_q_values)
#     return torch.gather(slates, max_q_slate_index, axis=0)
