from scripts.simulation_imports import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print("DEVICE: ", DEVICE)


def optimize_model(batch):
    optimizer.zero_grad()

    a = 5
    (
        state_batch,  # [batch_size, num_item_features]
        selected_doc_feat_batch,  # [batch_size, num_item_features]
        candidates_batch,  # [batch_size, num_candidates, num_item_features]
        reward_batch,  # [batch_size, 1]
        next_state_batch,  # [batch_size, num_item_features]
    ) = zip(*batch)

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

    ######## User related parameters ########
    state_model_cls = config["parameters"]["state_model_cls"]["value"]
    choice_model_cls = config["parameters"]["choice_model_cls"]["value"]
    response_model_cls = config["parameters"]["response_model_cls"]["value"]

    satisfaction_threshold = config["parameters"]["satisfaction_threshold"]["value"]
    resp_amp_factor = config["parameters"]["resp_amp_factor"]["value"]
    state_update_rate = config["parameters"]["state_update_rate"]["value"]

    ######## Environment related parameters ########
    SLATE_SIZE = config["parameters"]["slate_size"]["value"]
    NUM_CANDIDATES = config["parameters"]["num_candidates"]["value"]
    NUM_USERS = config["parameters"]["num_users"]["value"]
    NUM_ITEM_FEATURES = config["parameters"]["num_item_features"]["value"]
    RETRIEVAL_MODEL = config["parameters"]["retrieval_model"]["value"]
    INTENT_KIND = config["parameters"]["intent_kind"]["value"]

    ######## Training related parameters ########
    BATCH_SIZE = config["parameters"]["batch_size"]["value"]
    GAMMA = config["parameters"]["gamma"]["value"]
    TAU = config["parameters"]["tau"]["value"]
    LR = float(config["parameters"]["lr"]["value"])
    NUM_EPISODES = config["parameters"]["num_episodes"]["value"]

    ######## Models related parameters ########
    slate_gen_model_cls = config["parameters"]["slate_gen_model_cls"]["value"]
    GRU_SEQ_LEN = 10

    ##################################################
    #################### CATALOGUE ###################
    data_df = load_spotify_data()
    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")

    ################### RETRIEVAL MODEL ###################
    item_features = doc_catalogue.get_all_item_features()
    rec_model = ContentSimilarityRec(item_feature_matrix=item_features)

    ################## USER SAMPLER ###################
    feat_gen = NormalUserFeaturesGenerator()
    state_model_cls = class_name_to_class[state_model_cls]
    choice_model_cls = class_name_to_class[choice_model_cls]
    response_model_cls = class_name_to_class[response_model_cls]

    state_model_kwgs = {"state_update_rate": state_update_rate}
    choice_model_kwgs = {"satisfaction_threshold": satisfaction_threshold}
    response_model_kwgs = {"amp_factor": resp_amp_factor}

    user_sampler = UserSampler(
        feat_gen,
        state_model_cls,
        choice_model_cls,
        response_model_cls,
        state_model_kwargs=state_model_kwgs,
        choice_model_kwargs=choice_model_kwgs,
        response_model_kwargs=response_model_kwgs,
    )
    user_sampler.generate_users(num_users=NUM_USERS)

    # TODO: dont really now why needed there we shold use the one associated to the user sampled for the episode
    choice_model = choice_model_cls(satisfaction_threshold=satisfaction_threshold)

    env = MusicGym(
        user_sampler=user_sampler,
        doc_catalogue=doc_catalogue,
        rec_model=rec_model,
        k=NUM_CANDIDATES,
        device=DEVICE,
    )

    # define Slate Gen model
    slate_gen_model_cls = class_name_to_class[slate_gen_model_cls]
    slate_gen = slate_gen_model_cls(slate_size=SLATE_SIZE)

    # defining Belief Agent
    # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
    agent = DQNAgent(
        slate_gen=slate_gen, input_size=2 * NUM_ITEM_FEATURES, output_size=1
    )

    belief_model = GRUModel(num_doc_features=NUM_ITEM_FEATURES)

    bf_agent = BeliefAgent(agent=agent, belief_model=belief_model).to(device=DEVICE)

    transition_cls = GruTransition

    replay_memory_dataset = ReplayMemoryDataset(
        capacity=100_000, transition_cls=transition_cls
    )
    replay_memory_dataloader = DataLoader(
        replay_memory_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=replay_memory_dataset.collate_fn,
        shuffle=False,
    )

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(bf_agent.parameters(), lr=LR)

    is_terminal = False

    for i_episode in tqdm(range(NUM_EPISODES)):
        gru_buff = torch.zeros((1, GRU_SEQ_LEN, NUM_ITEM_FEATURES)).to(DEVICE)
        count = 0

        reward = []
        loss = []

        env.reset()
        # Initialize b_u
        b_u = torch.Tensor(env.curr_user.features).to(DEVICE)

        is_terminal = False
        cum_reward = 0

        candidate_docs = env.get_candidate_docs()
        candidate_docs_repr = torch.Tensor(
            env.doc_catalogue.get_docs_features(candidate_docs)
        ).to(DEVICE)

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

                selected_doc_feature, response, is_terminal, _, _ = env.step(
                    slate, indicator=True
                )

                # fill the GRU buffer
                gru_buff[
                    0, GRU_SEQ_LEN - (count % GRU_SEQ_LEN) - 1, :
                ] = selected_doc_feature

                # push memory
                replay_memory_dataset.push(
                    transition_cls(
                        b_u,
                        selected_doc_feature,
                        candidate_docs_repr,
                        response,
                        gru_buff,
                    )
                )

                # output of the GRU cell, get the last output for the sequence
                out = bf_agent.update_belief(gru_buff)
                b_u = out[0, -1, :]

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

        log_dit = {
            "cum_reward": ep_reward,
            "avg_reward": ep_avg_reward,
        }
        if len(replay_memory_dataset.memory) >= (1 * BATCH_SIZE):
            log_dit["loss"] = torch.mean(torch.tensor(loss))

        wandb.log(log_dit, step=i_episode)

        print(
            "Loss: {}, Reward: {}, Cum_Rew: {}".format(
                torch.mean(torch.tensor(loss)),
                ep_avg_reward,
                ep_reward,
            )
        )