from rl_recsys.user_modeling.features_gen import UniformFeaturesGenerator
from scripts.simulation_imports import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print("DEVICE: ", DEVICE)


def optimize_model(batch):
    optimizer.zero_grad()

    (
        state_batch,  # [batch_size, num_item_features]
        selected_doc_feat_batch,  # [batch_size, num_item_features]
        candidates_batch,  # [batch_size, num_candidates, num_item_features]
        reward_batch,  # [batch_size, 1]
        gru_buffer_batch,  # [batch_size, num_item_features]
    ) = batch

    # print(reward_batch)
    # Q(s, a): [batch_size, 1]
    q_val = bf_agent.agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )  # type: ignore

    # compute s'
    gru_out = bf_agent.update_belief(gru_buffer_batch)
    next_state_batch = gru_out[
        :, -1, :
    ]  # keep only the last gru output for every batch

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
        scores_tens = torch.softmax(scores_tens, dim=0)

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
    SEED = config["parameters"]["seed"]["value"]
    pl.seed_everything(SEED)

    ######## Models related parameters ########
    slate_gen_model_cls = config["parameters"]["slate_gen_model_cls"]["value"]
    GRU_SEQ_LEN = config["parameters"]["hist_length"]["value"]

    ##################################################
    #################### CATALOGUE ###################
    data_df = load_spotify_data()
    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")

    ################### RETRIEVAL MODEL ###################
    item_features = doc_catalogue.get_all_item_features()
    rec_model = ContentSimilarityRec(item_feature_matrix=item_features)

    ################## USER SAMPLER ###################
    feat_gen = UniformFeaturesGenerator()
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
        device=DEVICE,
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
        slate_gen=slate_gen, input_size=2 * NUM_ITEM_FEATURES, output_size=1, tau=TAU
    )

    belief_model = GRUModel(num_doc_features=NUM_ITEM_FEATURES)

    bf_agent = BeliefAgent(agent=agent, belief_model=belief_model).to(device=DEVICE)

    transition_cls = GruTransition

    replay_memory_dataset = ReplayMemoryDataset(
        capacity=10_000, transition_cls=transition_cls
    )
    replay_memory_dataloader = DataLoader(
        replay_memory_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=replay_memory_dataset.collate_fn,
        # shuffle=False,
    )

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(bf_agent.parameters(), lr=LR)

    is_terminal = False
    keys = [
        "ep_reward",
        "ep_avg_reward",
        "loss",
        "best_rl_avg_diff",
        "avg_avd_diff",
        "cum_normalized",
    ]
    save_dict = defaultdict(list)
    save_dict.update({key: [] for key in keys})

    for i_episode in tqdm(range(NUM_EPISODES)):
        gru_buff = torch.zeros((1, GRU_SEQ_LEN, NUM_ITEM_FEATURES)).to(DEVICE)
        count = 0

        reward = []
        loss = []
        diff_to_best = []

        max_sess = []
        avg_sess = []

        env.reset()
        # Initialize b_u
        b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
        # b_u = (torch.randn(NUM_ITEM_FEATURES) * 2 - 1).to(DEVICE)

        is_terminal = False

        candidate_docs = env.get_candidate_docs()
        candidate_docs_repr = torch.Tensor(
            env.doc_catalogue.get_docs_features(candidate_docs)
        ).to(DEVICE)

        while not is_terminal:
            with torch.no_grad():
                ##########################################################################
                max_sess.append(
                    torch.mm(
                        env.curr_user.get_state().unsqueeze(0),
                        candidate_docs_repr.t(),
                    )
                    .squeeze(0)
                    .max()
                )

                avg_sess.append(
                    torch.mm(
                        env.curr_user.get_state().unsqueeze(0),
                        candidate_docs_repr.t(),
                    )
                    .squeeze(0)
                    .mean()
                )
                ##########################################################################
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
                scores = torch.softmax(scores, dim=0)
                q_val = q_val.squeeze()

                slate = bf_agent.get_action(scores, q_val)

                selected_doc_feature, response, is_terminal, _, _ = env.step(slate)

                # fill the GRU buffer
                if count % GRU_SEQ_LEN == 0 and count != 0:
                    # shift the buffer
                    for i in range(GRU_SEQ_LEN - 1):
                        gru_buff[0, i, :] = gru_buff[0, i + 1, :]
                    gru_buff[0, -1, :] = selected_doc_feature
                else:
                    gru_buff[0, count % GRU_SEQ_LEN, :] = selected_doc_feature
                count += 1

                # print(gru_buff)
                # if count == 3:
                #     exit(0)

                # push memory
                replay_memory_dataset.push(
                    transition_cls(
                        b_u,
                        selected_doc_feature,
                        candidate_docs_repr,
                        response,
                        gru_buff.squeeze(),
                    )
                )

                # output of the GRU cell, get the last output for the sequence
                out = bf_agent.update_belief(gru_buff)
                b_u = out[0, -1, :]

                reward.append(response)

        # optimize model
        if len(replay_memory_dataset.memory) >= 1 * (BATCH_SIZE):
            # get a batch of transitions from the replay buffer
            batch = next(iter(replay_memory_dataloader))
            for elem in batch:
                elem.to(DEVICE)
            batch_loss = optimize_model(batch)
            bf_agent.agent.soft_update_target_network()

            # accumulate loss for each episode
            loss.append(batch_loss)

        ep_avg_reward = torch.mean(torch.tensor(reward))
        ep_cum_reward = torch.sum(torch.tensor(reward))

        loss = torch.mean(torch.tensor(loss))

        ep_max_avg = torch.mean(torch.tensor(max_sess))
        ep_max_cum = torch.sum(torch.tensor(max_sess))

        ep_avg_avg = torch.mean(torch.tensor(avg_sess))
        ep_avg_cum = torch.sum(torch.tensor(avg_sess))

        cum_normalized = ep_cum_reward / ep_max_cum

        print(
            "Loss: {}\n Avg_Reward: {} - Cum_Rew: {}\n Max_Avg_Reward: {} - Max_Cum_Rew: {}\n Avg_Avg_Reward: {} - Avg_Cum_Rew: {}: - Cumulative_Normalized: {}".format(
                loss,
                ep_avg_reward,
                ep_cum_reward,
                ep_max_avg,
                ep_max_cum,
                ep_avg_avg,
                ep_avg_cum,
                cum_normalized,
            )
        )

        log_dit = {
            "avg_reward": ep_avg_reward,
            "cum_reward": ep_cum_reward,
            "max_avg": ep_max_avg,
            "max_cum": ep_max_cum,
            "avg_avg": ep_avg_avg,
            "avg_cum": ep_avg_cum,
            "best_rl_avg_diff": ep_max_avg - ep_avg_reward,
            "best_avg_avg_diff": ep_max_avg - ep_avg_avg,
            "cum_normalized": cum_normalized,
        }

        if len(replay_memory_dataset.memory) >= 1 * (BATCH_SIZE):
            log_dit["loss"] = loss

        wandb.log(log_dit, step=i_episode)

        save_dict["ep_reward"].append(ep_cum_reward)
        save_dict["ep_avg_reward"].append(ep_avg_reward)
        save_dict["loss"].append(loss)
        save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_reward)
        save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
        save_dict["cum_normalized"].append(cum_normalized)
    now = datetime.now()
    folder_name = now.strftime("%m-%d_%H-%M-%S")
    directory = "saved_models/hidden_slateq/gru/"

    # Create the directory with the folder name
    path = directory + folder_name
    os.makedirs(path)

    source_path = "src/scripts/config.yaml"
    destination_path = path + "/_config.yaml"
    shutil.copy(source_path, destination_path)

    # Save the model
    Path = path + "/model.pt"
    torch.save(bf_agent, Path)

    with open(path + "/logs_dict.pickle", "wb") as f:
        pickle.dump(save_dict, f)
