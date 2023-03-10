import pytest

from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.user_modeling.choice_model import DotProductChoiceModel
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.user_modeling.response_model import DotProductResponseModel
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.utils import load_spotify_data


@pytest.fixture
def user_sampler():
    # object
    feat_gen = NormalUserFeaturesGenerator()
    # classes
    state_model_cls = AlphaIntentUserState
    choice_model_cls = DotProductChoiceModel
    response_model_cls = DotProductResponseModel

    user_sampler = UserSampler(
        feat_gen, state_model_cls, choice_model_cls, response_model_cls
    )
    user_sampler.generate_users(num_users=1000)
    return user_sampler


@pytest.fixture
def doc_catalogue():
    data_df = load_spotify_data()
    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")
    return doc_catalogue


@pytest.fixture
def content_rec(doc_catalogue):
    item_features = doc_catalogue.get_all_item_features()
    rec_model = ContentSimilarityRec(item_feature_matrix=item_features)
    return rec_model
