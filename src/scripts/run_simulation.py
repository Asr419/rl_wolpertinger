import argparse
import configparser

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from rl_recsys.agent_modeling.agent import BeliefAgent
from rl_recsys.agent_modeling.dqn_agent import (
    DQNAgent,
    ReplayMemoryDataset,
    replay_memory_collate_fn,
)
from rl_recsys.agent_modeling.slate_generator import TopKSlateGenerator
from rl_recsys.belief_modeling.history_model import AvgHistoryModel, GRUModel
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.simulation_environment.environment import MusicGym
from rl_recsys.user_modeling.choice_model import (
    CosineSimilarityChoiceModel,
    DotProductChoiceModel,
)
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.user_modeling.response_model import (
    CosineResponseModel,
    DotProductResponseModel,
)
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.utils import load_spotify_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="src/scripts/config.yaml",
    help="Path to the config file.",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)
wandb.init(project="rl_recsys", config=config["parameters"])

BATCH_SIZE = config["parameters"]["batch_size"]["value"]
GAMMA = config["parameters"]["gamma"]["value"]
TAU = config["parameters"]["tau"]["value"]
LR = float(config["parameters"]["lr"]["value"])
NUM_EPISODES = config["parameters"]["num_episodes"]["value"]
SLATE_SIZE = config["parameters"]["slate_size"]["value"]
NUM_ITEM_FEATURES = config["parameters"]["num_item_features"]["value"]
NUM_CANDIDATES = config["parameters"]["num_candidates"]["value"]
NUM_USERS = config["parameters"]["num_users"]["value"]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print("DEVICE: ", DEVICE)


def optimize_model(batch):
    optimizer.zero_grad()

    (
        state_batch,  # [batch_size, num_item_features]
        selected_doc_feat_batch,  # [batch_size, num_item_features]
        candidates_batch,  # [batch_size, num_candidates, num_item_features]
        reward_batch,  # [batch_size, 1]
        next_state_batch,  # [batch_size, num_item_features]
    ) = batch

    # Q(s, a): [batch_size, 1]
    q_val = bf_agent.agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )  # type: ignore

    # Q(s', a): [batch_size, 1]
    cand_qtgt_list = []
    for b in range(next_state_batch.shape[0]):
        next_state = next_state_batch[b, :]
        candidates = candidates_batch[b, :, :]

        next_state_rep = next_state.repeat((candidates.shape[0], 1))
        cand_qtgt = bf_agent.agent.compute_q_values(
            next_state_rep, candidates, use_policy_net=False
        )  # type: ignore

        choice_model.score_documents(next_state, candidates)
        # [num_candidates, 1]
        scores_tens = torch.Tensor(choice_model.scores).to(DEVICE).unsqueeze(dim=1)
        # max over Q(s', a)
        cand_qtgt_list.append((cand_qtgt * scores_tens).max())

    q_tgt = torch.stack(cand_qtgt_list).unsqueeze(dim=1)
    expected_q_values = q_tgt * GAMMA + reward_batch.unsqueeze(dim=1)
    loss = criterion(q_val, expected_q_values)

    # Optimize the model
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    #################### CATALOGUE ###################
    data_df = load_spotify_data()
    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")

    ################### RETRIEVAL MODEL ###################
    item_features = doc_catalogue.get_all_item_features()
    rec_model = ContentSimilarityRec(item_feature_matrix=item_features)

    ################## USER SAMPLER ###################
    feat_gen = NormalUserFeaturesGenerator()
    class_name_to_class = {
        "AlphaIntentUserState": AlphaIntentUserState,
        "DotProductChoiceModel": DotProductChoiceModel,
        "CosineResponseModel": CosineResponseModel,
        "CosineSimilarityChoiceModel": CosineSimilarityChoiceModel,
        "DotProductResponseModel": DotProductResponseModel,
    }

    state_model_cls = config["parameters"]["state_model_cls"]["value"]
    state_model_cls = class_name_to_class[state_model_cls]

    choice_model_cls = config["parameters"]["choice_model_cls"]["value"]
    choice_model_cls = class_name_to_class[choice_model_cls]

    response_model_cls = config["parameters"]["response_model_cls"]["value"]
    response_model_cls = class_name_to_class[response_model_cls]

    satisfaction_threshold = config["parameters"]["satisfaction_threshold"]["value"]
    choice_model_class = config["parameters"]["choice_model"]["value"]
    choice_model = class_name_to_class[choice_model_class](
        satisfaction_threshold=satisfaction_threshold
    )

    satisfaction_threshold = config["parameters"]["satisfaction_threshold"]["value"]

    user_sampler = UserSampler(
        feat_gen, state_model_cls, choice_model_cls, response_model_cls
    )
    user_sampler.generate_users(num_users=NUM_USERS)

    env = MusicGym(
        user_sampler=user_sampler,
        doc_catalogue=doc_catalogue,
        rec_model=rec_model,
        k=NUM_CANDIDATES,
        device=DEVICE,
    )

    # define Slate Gen model
    slate_gen = TopKSlateGenerator(slate_size=SLATE_SIZE)

    # defining Belief Agent
    # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
    agent = DQNAgent(
        slate_gen=slate_gen, input_size=2 * NUM_ITEM_FEATURES, output_size=1
    )
    belief_model = AvgHistoryModel(num_doc_features=NUM_ITEM_FEATURES)
    # belief_model = GRUModel(
    #     num_doc_features=NUM_ITEM_FEATURES, hidden_size=14, output_size=14, num_layers=3
    # )
    bf_agent = BeliefAgent(agent=agent, belief_model=belief_model).to(device=DEVICE)

    replay_memory_dataset = ReplayMemoryDataset(capacity=100_000)
    replay_memory_dataloader = DataLoader(
        replay_memory_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=replay_memory_collate_fn,
        shuffle=False,
    )

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(bf_agent.parameters(), lr=LR)

    is_terminal = False
    b_u = None

    for i_episode in tqdm(range(NUM_EPISODES)):
        reward = []
        loss = []

        env.reset()
        is_terminal = False
        cum_reward = 0

        candidate_docs = env.get_candidate_docs()
        candidate_docs_repr = torch.Tensor(
            env.doc_catalogue.get_docs_features(candidate_docs)
        ).to(DEVICE)

        # initialize b_u with user features
        b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
        # b_u = torch.randn(14).to(DEVICE)

        while not is_terminal:
            with torch.no_grad():
                b_u_rep = b_u.repeat((candidate_docs_repr.shape[0], 1))

                q_val = bf_agent.agent.compute_q_values(
                    state=b_u_rep,
                    candidate_docs_repr=candidate_docs_repr,
                    use_policy_net=True,
                )  # type: ignore

                choice_model.score_documents(
                    user_state=b_u, docs_repr=candidate_docs_repr
                )
                scores = torch.Tensor(choice_model.scores).to(DEVICE)
                q_val = q_val.squeeze()
                slate = bf_agent.get_action(scores, q_val)

                selected_doc_feature, response, is_terminal, _, _ = env.step(slate, b_u)

                b_u_next = bf_agent.update_belief(selected_doc_feature)

                # push memory
                replay_memory_dataset.push(
                    b_u,
                    selected_doc_feature,
                    candidate_docs_repr,
                    response,
                    b_u_next,
                )
                b_u = b_u_next
                # accumulate reward for each episode
                # print(response)
                reward.append(response)

            # optimize model
            if len(replay_memory_dataset.memory) >= 1 * BATCH_SIZE:
                # get a batch of transitions from the replay buffer
                batch = next(iter(replay_memory_dataloader))
                for elem in batch:
                    elem.to(DEVICE)
                batch_loss = optimize_model(batch)
                bf_agent.agent.soft_update_target_network()

                # accumulate loss for each episode
                loss.append(batch_loss)

        ep_avg_reward = torch.mean(torch.tensor(reward))
        ep_reward = torch.sum(torch.tensor(reward))

        log_dit = {"cum_reward": ep_reward, "avg_reward": ep_avg_reward}
        if len(replay_memory_dataset.memory) >= (1 * BATCH_SIZE):
            log_dit["loss"] = torch.mean(torch.tensor(loss))

        wandb.log(log_dit, step=i_episode)

        print(
            "Loss: {}, Reward: {}, Cum_Rew: {}".format(
                torch.mean(torch.tensor(loss)),
                torch.mean(torch.tensor(reward)),
                torch.sum(torch.tensor(reward)),
            )
        )
