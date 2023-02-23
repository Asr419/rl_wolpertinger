import numpy as np

from rl_recsys.user_modeling.user_state import AlphaIntentUserState


def test_alpha_intent_user_state():
    """Test alpha intent user state"""
    num_features = 5
    user_features = np.random.uniform(0, 1, num_features)
    alph_state = AlphaIntentUserState(user_features=user_features)
    assert alph_state.user_state.shape == user_features.shape
    assert all(user_features != alph_state.user_state)
    alph_state.update_state()
