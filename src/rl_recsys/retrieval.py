from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentSimilarityRec:
    # implement the retrieval model
    item_feature_matrix: npt.NDArray[np.float64]

    def recommend(
        self,
        user_features: npt.NDArray[np.float64],
        k=10,
    ) -> npt.NDArray[np.int_]:
        # compute the similarity between the user and the items
        # and return the top k items
        scores = np.dot(user_features, self.item_feature_matrix.T)
        norms = np.linalg.norm(user_features) * np.linalg.norm(
            self.item_feature_matrix, axis=1
        )
        cos_scores = scores / norms
        top_k_items = np.argsort(cos_scores)[-k:]
        print(cos_scores.max())
        return np.array(top_k_items.tolist())

    def recommend_random(
        self,
        user_features: npt.NDArray[np.float64],
        k=10,
    ) -> npt.NDArray[np.int_]:
        seed = hash(tuple(user_features)) % (2**32 - 1)
        np.random.seed(seed)
        top_k_items = np.random.choice(self.item_feature_matrix.shape[0], k)
        return np.array(top_k_items.tolist())


class Random_Recommender:
    item_feature_matrix: npt.NDArray[np.float64]

    def recommend_random(
        self,
        user_features: npt.NDArray[np.float64],
        k=10,
    ) -> npt.NDArray[np.int_]:
        seed = hash(tuple(user_features))
        np.random.seed(seed)
        top_k_items = np.random.choice(self.item_feature_matrix.shape[0], k)
        return np.array(top_k_items.tolist())
