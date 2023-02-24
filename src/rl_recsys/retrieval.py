from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentSimilarityRec:
    # implement the retrieval model
    item_feature_matrix: npt.NDArray[np.float64]
    similarity_mat: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.similarity_mat = cosine_similarity(self.item_feature_matrix)

    def recommend(
        self,
        user_repr: npt.NDArray[np.float64],
        k=10,
    ) -> list[int]:
        # compute the similarity between the user and the items
        # and return the top k items
        scores = np.dot(user_repr, self.similarity_mat)
        top_k_items = np.argsort(scores)[-k:]
        return top_k_items.tolist()
