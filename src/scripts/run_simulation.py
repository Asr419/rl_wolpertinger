from rl_recsys.agent_modeling.agent import Rl_agent
from rl_recsys.agent_modeling.slate_generator import Topk_slate
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.simulation_environment.environment import MusicGym
from rl_recsys.user_modeling.choice_model import DotProductChoiceModel
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.user_modeling.response_model import DotProductResponseModel
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.utils import load_spotify_data

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

NUM_EPISODES = 10

if __name__ == "__main__":
    # CATALOGUE
    data_df = load_spotify_data()
    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")

    # RETRIEVAL MODEL
    item_features = doc_catalogue.get_all_item_features()
    rec_model = ContentSimilarityRec(item_feature_matrix=item_features)

    # USER SAMPLER
    feat_gen = NormalUserFeaturesGenerator()

    state_model_cls = AlphaIntentUserState
    choice_model_cls = DotProductChoiceModel
    response_model_cls = DotProductResponseModel

    user_sampler = UserSampler(
        feat_gen, state_model_cls, choice_model_cls, response_model_cls
    )
    user_sampler.generate_users(num_users=100)

    env = MusicGym(
        user_sampler=user_sampler,
        doc_catalogue=doc_catalogue,
        rec_model=rec_model,
        k=100,
    )

    for i_episode in range(NUM_EPISODES):
        env.reset()

        state = env.get_curr_state()
        candidate_docs = env.get_candidate_docs()

        agent = ...
        belief_model = ...
        bf_agent = BeliefAgent(agent=agent, belief_model=belief_model)

        slate_gen_func = Topk_slate

        slate = Rl_agent(slate_gen_func, state, candidate_docs)
