from rl_recsys.user_modeling.choice_model import DeterministicChoicheModel
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.user_modeling.response_model import DotProductResponseModel
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState


def test_user_sampler():
    # check if user sampler working correctly
    # object
    feat_gen = NormalUserFeaturesGenerator()
    # classes
    state_model_cls = AlphaIntentUserState
    choice_model_cls = DeterministicChoicheModel
    response_model_cls = DotProductResponseModel

    sampler = UserSampler(
        feat_gen, state_model_cls, choice_model_cls, response_model_cls
    )

    sampler.generate_users()
    assert len(sampler.users) == 100
    user_i = sampler.sample_user()
