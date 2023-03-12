import torch
import torch.optim as optim

from rl_recsys.agent_modeling.agent import BeliefAgent
from rl_recsys.agent_modeling.dqn_agent import DQNAgent, ReplayMemory, Transition
from rl_recsys.agent_modeling.slate_generator import TopKSlateGenerator
from rl_recsys.belief_modeling.history_model import AvgHistoryModel
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.simulation_environment.environment import MusicGym
from rl_recsys.user_modeling.choice_model import DotProductChoiceModel
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.user_modeling.response_model import DotProductResponseModel
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.utils import load_spotify_data

BATCH_SIZE = 2
GAMMA = 1.0
TAU = 0.005
LR = 1e-4

NUM_EPISODES = 10

SLATE_SIZE = 10

NUM_ITEM_FEATURES = 14

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"


def optimize_model(transitions_batch):
    optimizer.zero_grad()
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions_batch))

    # [batch_size, num_features]
    selected_doc_feat_batch = torch.stack(batch.selected_doc_feat)
    state_batch = torch.stack(batch.state)
    # [batch_size, num_features]
    next_state_batch = torch.stack(batch.next_state)
    # [batch_size, candidates, num_features]
    candidates_batch = torch.stack(batch.candidates_docs)
    # [batch_size, reward]
    reward_batch = torch.stack(batch.reward)

    # Q(s, a): [batch_size, 1]
    q_val = bf_agent.agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )

    q_tgt = []
    for b in range(len(transitions_batch)):
        next_state = next_state_batch[b, :]
        candidates = candidates_batch[b, :, :]
        next_state_rep = next_state.repeat((candidates.shape[0], 1))
        q_tgt.append(
            bf_agent.agent.compute_q_values(
                next_state_rep, candidates, use_policy_net=False
            )
        )

    scores_list = []
    for b in range(len(transitions_batch)):
        choice_model.score_documents(state_batch[b, :], candidates_batch[b, :, :])
        scores_tens = torch.Tensor(choice_model.scores, device=DEVICE)
        scores_list.append(scores_tens)

    scores = torch.stack(scores_list)

    # Q(s', a'): [batch_size, candidates, 1]
    expected_q_values = (
        torch.stack(q_tgt).squeeze() * scores
    ) * GAMMA + reward_batch.squeeze()

    loss = criterion(q_val, expected_q_values)

    # Optimize the model
    loss.backward()
    optimizer.step()
    return loss


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

    choice_model = DotProductChoiceModel()

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

    # define Slate Gen model
    slate_gen = TopKSlateGenerator(slate_size=SLATE_SIZE)

    # defining Belief Agent
    # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
    agent = DQNAgent(
        slate_gen=slate_gen, input_size=2 * NUM_ITEM_FEATURES, output_size=1
    )
    belief_model = AvgHistoryModel(num_doc_features=NUM_ITEM_FEATURES)
    bf_agent = BeliefAgent(agent=agent, belief_model=belief_model)

    replay_memory = ReplayMemory(capacity=10_000)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(bf_agent.parameters(), lr=LR)

    is_terminal = False
    b_u = None

    for i_episode in range(NUM_EPISODES):
        env.reset()
        candidate_docs = env.get_candidate_docs()
        candidate_docs_repr = env.doc_catalogue.get_docs_features(candidate_docs)
        # initialize b_u with user features
        b_u = env.curr_user.features

        b_u_tens = torch.Tensor(b_u).to(DEVICE)
        candidate_docs_repr_tens = torch.Tensor(candidate_docs_repr).to(DEVICE)

        while not is_terminal:
            with torch.no_grad():
                # Todo: np array into tensor

                b_u_rep = b_u_tens.repeat((candidate_docs_repr_tens.shape[0], 1))

                q_val = bf_agent.agent.compute_q_values(
                    state=b_u_rep,
                    candidate_docs_repr=candidate_docs_repr_tens,
                    use_policy_net=True,
                )  # type: ignore

                choice_model.score_documents(
                    user_state=b_u, docs_repr=candidate_docs_repr
                )
                scores = torch.Tensor(choice_model.scores).to(DEVICE)

                q_val = q_val.squeeze()
                slate = bf_agent.get_action(scores, q_val)
                selected_doc_feature, response, is_terminal, _, _ = env.step(slate, b_u)

                selected_doc_feature = torch.Tensor(selected_doc_feature).to(DEVICE)
                response = torch.Tensor([response]).to(DEVICE)

                b_u_next = bf_agent.update_belief(selected_doc_feature)

                # push memory
                replay_memory.push(
                    b_u_tens,
                    selected_doc_feature,
                    response,
                    b_u_next,
                    candidate_docs_repr_tens,
                )
                b_u = b_u_next

            # optimize model
            if len(replay_memory.memory) >= BATCH_SIZE:
                # sample a batch of transitions from the replay buffer
                transitions_batch = replay_memory.sample(BATCH_SIZE)
                optimize_model(transitions_batch)

                bf_agent.agent.soft_update_target_network()
