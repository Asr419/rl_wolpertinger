from scripts.simulation_imports import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print("DEVICE: ", DEVICE)
PATH = "src/saved_models/random_slateq/04-08_21-00-37/model.pt"

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
    keys = ["ep_reward", "ep_avg_reward", "best_rl_avg_diff", "avg_avd_diff"]
    save_dict = defaultdict(list)
    save_dict.update({key: [] for key in keys})
   

    for i_episode in tqdm(range(NUM_EPISODES)):
        gru_buff = torch.zeros((1, GRU_SEQ_LEN, NUM_ITEM_FEATURES)).to(DEVICE)
        count = 0

        reward = []
        
        diff_to_best = []

        env.reset()
        # Initialize b_u
        b_u = torch.Tensor(env.curr_user.features).to(DEVICE)

        is_terminal = False
        cum_reward = 0

        candidate_docs = env.get_candidate_docs()
        candidate_docs_repr = torch.Tensor(
            env.doc_catalogue.get_docs_features(candidate_docs)
        ).to(DEVICE)

        print("NEW EPISODE")
        cos_sim = torch.nn.functional.cosine_similarity(
            env.curr_user.get_state(), candidate_docs_repr, dim=1
        )
        print(cos_sim.max())
        print(cos_sim.min())
        print(cos_sim.mean())
        print("++++++++")
        max_sess = []
        avg_sess = []
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

                selected_doc_feature, response, is_terminal, _, _ = env.step(slate)
                
                

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
                        gru_buff.squeeze(),
                    )
                )

                # output of the GRU cell, get the last output for the sequence
                out = bf_agent.update_belief(gru_buff)
                b_u = out[0, -1, :]

                reward.append(response)
        ep_avg_reward = torch.mean(torch.tensor(reward))
        ep_cum_reward = torch.sum(torch.tensor(reward))

        

        ep_max_avg = torch.mean(torch.tensor(max_sess))
        ep_max_cum = torch.sum(torch.tensor(max_sess))

        ep_avg_avg = torch.mean(torch.tensor(avg_sess))
        ep_avg_cum = torch.sum(torch.tensor(avg_sess))

        print(
            "Avg_Reward: {} - Cum_Rew: {}\n Max_Avg_Reward: {} - Max_Cum_Rew: {}\n Avg_Avg_Reward: {} - Avg_Cum_Rew: {}:".format(
                ep_avg_reward,
                ep_cum_reward,
                ep_max_avg,
                ep_max_cum,
                ep_avg_avg,
                ep_avg_cum,
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
        }

    


        wandb.log(log_dit, step=i_episode)

        save_dict["ep_reward"].append(ep_cum_reward)
        save_dict["ep_avg_reward"].append(ep_avg_reward)
       
        save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_reward)
        save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)