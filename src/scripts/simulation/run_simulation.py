from rl_recsys.user_modeling.features_gen import UniformFeaturesGenerator
from scripts.simulation_imports import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print("DEVICE: ", DEVICE)


def update_belief(belief_state:torch.Tensor,selected_doc_feature: torch.Tensor, intent_kind: str):
    b_u_next = None
    if intent_kind == "random":
        # create a randm tensor between -1 and 1
        b_u_next = torch.randn(14) * 2 - 1
    if intent_kind == "static":
        b_u_next = belief_state
    if intent_kind == "observable":
        b_u_next = env.curr_user.get_state()
    if intent_kind=="hidden":
        b_u_next = bf_agent.update_belief(selected_doc_feature)
    return b_u_next


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

    # with torch.no_grad():
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
        scores_tens = torch.softmax(scores_tens, dim=0)

        cand_qtgt_list.append((cand_qtgt * scores_tens).max())

    q_tgt = torch.stack(cand_qtgt_list).unsqueeze(dim=1)
    expected_q_values = q_tgt * GAMMA + reward_batch.unsqueeze(dim=1)
    # expected_q_values = q_tgt
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
    history_model_cls = config["parameters"]["history_model_cls"]["value"]
    slate_gen_model_cls = config["parameters"]["slate_gen_model_cls"]["value"]
    SEQ_LEN = config["parameters"]["hist_length"]["value"]

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
        slate_gen=slate_gen, input_size=2 * NUM_ITEM_FEATURES, output_size=1,tau=TAU
    )

    history_model_cls = class_name_to_class[history_model_cls]
    belief_model = history_model_cls(
        num_doc_features=NUM_ITEM_FEATURES, memory_length=SEQ_LEN
    )

    bf_agent = BeliefAgent(agent=agent, belief_model=belief_model).to(device=DEVICE)
    transition_cls = Transition

    replay_memory_dataset = ReplayMemoryDataset(
        capacity=10_000, transition_cls=transition_cls
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
    # Initialize b_u
    b_u = None

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
        reward = []
        loss = []
        diff_to_best = []

        env.reset()
        is_terminal = False
        cum_reward = 0

        candidate_docs = env.get_candidate_docs()
        candidate_docs_repr = torch.Tensor(
            env.doc_catalogue.get_docs_features(candidate_docs)
        ).to(DEVICE)

        if INTENT_KIND == "random":
            b_u = (torch.randn(14) * 2 - 1).to(DEVICE)
        elif INTENT_KIND == "static":
            b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
        elif INTENT_KIND == "observable":
            b_u = torch.Tensor(env.curr_user.get_state()).to(DEVICE)
        elif INTENT_KIND=="hidden":
            b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
        else:
            raise ValueError("invalid intent_kind")

        max_sess = []
        avg_sess = []
        # print(env.curr_user.get_state())
        while not is_terminal:
            with torch.no_grad():
                ##########################################################################
                max_sess.append(
                    torch.mm(
                        env.curr_user.get_state().unsqueeze(0), candidate_docs_repr.t()
                    )
                    .squeeze(0)
                    .max()
                )

                avg_sess.append(
                    torch.mm(
                        env.curr_user.get_state().unsqueeze(0), candidate_docs_repr.t()
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
                # print(slate)

                selected_doc_feature, response, is_terminal, _, _ = env.step(slate)
                # print(response)

                b_u_next = update_belief(belief_state=b_u,
                    selected_doc_feature=selected_doc_feature,
                    intent_kind=INTENT_KIND,
                )

                # push memory
                replay_memory_dataset.push(
                    transition_cls(
                        b_u,
                        selected_doc_feature,
                        candidate_docs_repr,
                        response,
                        b_u_next,
                    )
                )
                # print(b_u)
                b_u = b_u_next

                # print(response)
                reward.append(response)

            # optimize model
        if len(replay_memory_dataset.memory) >= 10 * BATCH_SIZE:
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
            "cum_normalized" :cum_normalized,
        }

        if len(replay_memory_dataset.memory) >= (1 * BATCH_SIZE):
            log_dit["loss"] = loss

        wandb.log(log_dit, step=i_episode)

        save_dict["ep_cum_reward"].append(ep_cum_reward)
        save_dict["ep_avg_reward"].append(ep_avg_reward)
        save_dict["loss"].append(loss)
        save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_reward)
        save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
        save_dict["cum_normalized"].append(cum_normalized)
    now = datetime.now()
    folder_name = now.strftime("%m-%d_%H-%M-%S")
    if INTENT_KIND == "random":
        directory = "saved_models/random_slateq/"
    elif INTENT_KIND == "static":
        directory = "saved_models/static/"
    elif INTENT_KIND == "observable":
        directory = "saved_models/observed_slateq/"

    # Create the directory with the folder name
    path = directory + folder_name
    os.makedirs(path)

    source_path = "src/scripts/config.yaml"
    destination_path = path + "/_config.yaml"
    shutil.copy(source_path, destination_path)

    # Save the model
    Path = path + f"/model_{SEED}.pt"
    torch.save(bf_agent, Path)

    with open(path + "/logs_dict.pickle", "wb") as f:
        pickle.dump(save_dict, f)
