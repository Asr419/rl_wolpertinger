from scripts.simulation_imports import *
from rl_recsys.user_modeling.features_gen import UniformFeaturesGenerator

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print("DEVICE: ", DEVICE)
PATH = "../../pomdp_saved_models/observed_slateq_11_04-18_22-47-20/model.pt"



def update_belief(belief_state: torch.Tensor,selected_doc_feature: torch.Tensor, intent_kind: str):
    b_u_next = None
    if intent_kind == "random":
        b_u_next = torch.randn(14)
    if intent_kind == "hidden":
        b_u_next = bf_agent.update_belief(selected_doc_feature)
    if intent_kind == "observable":
        b_u_next = env.curr_user.get_state()
    return b_u_next


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../../pomdp_saved_models/observed_slateq_11_04-18_22-47-20/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # wandb.init(project="rl_recsys", config=config["parameters"])
    SEEDS = [23,25,27,29]
    for seed in SEEDS:
        pl.seed_everything(seed)
    ######## User related parameters ########
        state_model_cls = config["parameters"]["state_model_cls"]["value"]
        choice_model_cls = config["parameters"]["choice_model_cls"]["value"]
        response_model_cls = config["parameters"]["response_model_cls"]["value"]

        resp_amp_factor = config["parameters"]["resp_amp_factor"]["value"]
        state_update_rate = config["parameters"]["state_update_rate"]["value"]

        ######## Environment related parameters ########
        SLATE_SIZE = config["parameters"]["slate_size"]["value"]
        NUM_CANDIDATES = config["parameters"]["num_candidates"]["value"]
        NUM_USERS = config["parameters"]["num_users"]["value"]
        NUM_ITEM_FEATURES = config["parameters"]["num_item_features"]["value"]
        SONG_PER_SESSION = config["parameters"]["song_per_session"]["value"]
        INTENT_KIND = config["parameters"]["intent_kind"]["value"]

        ######## Training related parameters ########
        BATCH_SIZE = config["parameters"]["batch_size"]["value"]
        GAMMA = config["parameters"]["gamma"]["value"]
        TAU = config["parameters"]["tau"]["value"]
        LR = float(config["parameters"]["lr"]["value"])
        NUM_EPISODES = config["parameters"]["num_episodes"]["value"]

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
        user_feat_gen = UniformFeaturesGenerator()
        intent_feat_gen = UniformFeaturesGenerator()
        state_model_cls = class_name_to_class[state_model_cls]
        choice_model_cls = class_name_to_class[choice_model_cls]
        response_model_cls = class_name_to_class[response_model_cls]

        state_model_kwgs = {
                "state_update_rate": state_update_rate,
                "intent_gen": intent_feat_gen,
            }
        choice_model_kwgs = {}
        response_model_kwgs = {"amp_factor": resp_amp_factor}

        user_sampler = UserSampler(
                user_feat_gen,
                state_model_cls,
                choice_model_cls,
                response_model_cls,
                state_model_kwargs=state_model_kwgs,
                choice_model_kwargs=choice_model_kwgs,
                response_model_kwargs=response_model_kwgs,
                songs_per_sess=SONG_PER_SESSION,
                device=DEVICE,
            )
        user_sampler.generate_users(num_users=NUM_USERS)

        # TODO: dont really now why needed there we shold use the one associated to the user sampled for the episode
        choice_model = choice_model_cls()

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

        history_model_cls = class_name_to_class[history_model_cls]
        belief_model = history_model_cls(
            num_doc_features=NUM_ITEM_FEATURES, memory_length=SEQ_LEN
        )

        # bf_agent = BeliefAgent(agent=agent, belief_model=belief_model).to(device=DEVICE)
        bf_agent = torch.load(PATH)

        is_terminal = False
        # Initialize b_u
        b_u = None

        # keys = ["ep_reward", "ep_avg_reward"]
        # save_dict = defaultdict(list)
        # save_dict.update({key: [] for key in keys})
        now = datetime.now()
        time_now = now.strftime("%m-%d_%H-%M-%S")
        RUN_NAME = f"Observable_SEED_{11}_{time_now}"
        wandb.init(project="rl_recsys", config=config["parameters"], name=RUN_NAME)
        for i_episode in tqdm(range(NUM_EPISODES)):
            reward = []
            

            env.reset()
            is_terminal = False
            cum_reward = 0

            candidate_docs = env.get_candidate_docs()
            candidate_docs_repr = torch.Tensor(
                env.doc_catalogue.get_docs_features(candidate_docs)
            ).to(DEVICE)
            
            if INTENT_KIND == "random":
                b_u = torch.randn(14).to(DEVICE)
            elif INTENT_KIND == "hidden":
                b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
            elif INTENT_KIND == "observable":
                b_u = torch.Tensor(env.curr_user.get_state()).to(DEVICE)
                
            else:
                raise ValueError("invalid intent_kind")
            print(INTENT_KIND)
            max_sess = []
            avg_sess = []
            while not is_terminal:
                with torch.no_grad():
                    # cos_sim = torch.nn.functional.cosine_similarity(
                    #     env.curr_user.get_state(), candidate_docs_repr, dim=1
                    # )
                    # print(cos_sim.max())
                    # print(cos_sim.min())
                    # print(cos_sim.mean())
                    # print("++++++++")
                    rew_cand = torch.mm(
                            env.curr_user.get_state().unsqueeze(0),
                            candidate_docs_repr.t(),
                        ).squeeze(0)
                    max_rew = rew_cand.max()
                    min_rew = rew_cand.min()
                    mean_rew = rew_cand.mean()
                    std_rew = rew_cand.std()

                    max_sess.append(max_rew)
                    avg_sess.append(mean_rew)
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

                    
                    response = (response - min_rew) / (max_rew - min_rew)
                    b_u_next = update_belief(
                        belief_state=b_u,
                        selected_doc_feature=selected_doc_feature,
                        intent_kind=INTENT_KIND,
                    )
                    reward.append(response)

                # optimize model

            ep_avg_reward = torch.mean(torch.tensor(reward))
            ep_cum_reward = torch.sum(torch.tensor(reward))

    

            ep_max_avg = torch.mean(torch.tensor(max_sess))
            ep_max_cum = torch.sum(torch.tensor(max_sess))

            ep_avg_avg = torch.mean(torch.tensor(avg_sess))
            ep_avg_cum = torch.sum(torch.tensor(avg_sess))

            if ep_max_cum > 0:
                cum_normalized = ep_cum_reward / ep_max_cum
            else:
                cum_normalized = ep_max_cum / ep_cum_reward

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

            
            wandb.log(log_dit, step=i_episode)
            # wandb.log(log_dit, step=i_episode)

            print(
                    "Avg_Reward: {} - Cum_Rew: {}\n Max_Avg_Reward: {} - Max_Cum_Rew: {}\n Avg_Avg_Reward: {} - Avg_Cum_Rew: {}: - Cumulative_Normalized: {}".format(
                        ep_avg_reward,
                        ep_cum_reward,
                        ep_max_avg,
                        ep_max_cum,
                        ep_avg_avg,
                        ep_avg_cum,
                        cum_normalized,
                    )
                )
        wandb.finish()
            
