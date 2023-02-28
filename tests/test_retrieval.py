from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.utils import load_spotify_data


def test_content_sim_rec():
    num_features = 14
    user_features_gen = NormalUserFeaturesGenerator()
    user_features = user_features_gen(num_features)

    spot_data = load_spotify_data()
    doc_catalogue = DocCatalogue(doc_df=spot_data, doc_id_column="song_id")
    item_features = doc_catalogue.get_all_item_features()

    rec_model = ContentSimilarityRec(item_feature_matrix=item_features)
    recs = rec_model.recommend(user_features=user_features, k=10)
    assert len(recs) == 10
