from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator


def test_normal_user_features_gen():
    """Test normal user features generator"""
    num_features = 5
    user_features_gen = NormalUserFeaturesGenerator()
    user_features = user_features_gen(num_features)
    assert user_features.shape == (num_features,)
