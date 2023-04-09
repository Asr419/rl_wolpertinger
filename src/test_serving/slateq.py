from scripts.simulation_imports import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print("DEVICE: ", DEVICE)
PATH = "src/saved_models/random_slateq/04-08_21-00-37/model.pt"


def update_belief(selected_doc_feature: torch.Tensor, intent_kind: str):
    b_u_next = None
    if intent_kind == "random":
        b_u_next = torch.randn(14)
    if intent_kind == "hidden":
        b_u_next = bf_agent.update_belief(selected_doc_feature)
    if intent_kind == "observable":
        b_u_next = env.curr_user.state_model.update_state(selected_doc_feature)
    return b_u_next


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/aayush/git/rl_recsys/src/saved_models/random_slateq/04-08_21-00-37/_config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # wandb.init(project="rl_recsys", config=config["parameters"])

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
    history_model_cls = config["parameters"]["history_model_cls"]["value"]
    slate_gen_model_cls = config["parameters"]["slate_gen_model_cls"]["value"]

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

    history_model_cls = class_name_to_class[history_model_cls]
    belief_model = history_model_cls(num_doc_features=NUM_ITEM_FEATURES)

    # bf_agent = BeliefAgent(agent=agent, belief_model=belief_model).to(device=DEVICE)
    bf_agent = torch.load(PATH)

    is_terminal = False
    # Initialize b_u
    b_u = None

    keys = ["ep_reward", "ep_avg_reward"]
    save_dict = defaultdict(list)
    save_dict.update({key: [] for key in keys})

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
        if INTENT_KIND == "random":
            b_u = torch.randn(14).to(DEVICE)
        elif INTENT_KIND == "hidden":
            b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
        elif INTENT_KIND == "observable":
            b_u = torch.Tensor(env.curr_user.get_state()).to(DEVICE)
        else:
            raise ValueError("invalid intent_kind")

        while not is_terminal:
            with torch.no_grad():
                # cos_sim = torch.nn.functional.cosine_similarity(
                #     env.curr_user.get_state(), candidate_docs_repr, dim=1
                # )
                # print(cos_sim.max())
                # print(cos_sim.min())
                # print(cos_sim.mean())
                # print("++++++++")

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

                if torch.any(selected_doc_feature != 0):
                    b_u_next = update_belief(
                        selected_doc_feature=selected_doc_feature,
                        intent_kind=INTENT_KIND,
                    )
                else:
                    # no item selected -> no update
                    b_u_next = b_u

                b_u = b_u_next

                reward.append(response)

            # optimize model

        ep_avg_reward = torch.mean(torch.tensor(reward))
        ep_reward = torch.sum(torch.tensor(reward))

        log_dit = {
            "cum_reward": ep_reward,
            "avg_reward": ep_avg_reward,
        }

        save_dict["ep_reward"].append(ep_reward)
        save_dict["ep_avg_reward"].append(ep_avg_reward)

        # wandb.log(log_dit, step=i_episode)

        print(
            "Loss: {}, Reward: {}, Cum_Rew: {}".format(
                torch.mean(torch.tensor(loss)),
                ep_avg_reward,
                ep_reward,
            )
        )
