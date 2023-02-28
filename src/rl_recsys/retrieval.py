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
        top_k_items = np.argsort(scores)[-k:]
        return np.array(top_k_items.tolist())
