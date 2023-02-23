from rl_recsys.user_modeling.user_features_gen import NormalUserFeaturesGenerator


def test_normal_user_features_gen():
    """Test normal user features generator"""
    num_features = 5
    mean = 0.0
    std = 1.0
    user_features_gen = NormalUserFeaturesGenerator(mean, std)
    user_features = user_features_gen(num_features)
    assert user_features.shape == (num_features,)
