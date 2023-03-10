import numpy as np

from rl_recsys.user_modeling.choice_model import DotProductChoiceModel


def test_dot_product_choie_model():
    """Test dot product choice model."""
    user_state = np.array([1, 2, 3])
    docs_repr = np.array([[1, 2, 3], [4, 5, 6]])
    choice_model = DotProductChoiceModel()
    choice_model.score_documents(user_state, docs_repr)

    assert choice_model.scores is not None
    assert choice_model.choose_document() in [0, 1]
