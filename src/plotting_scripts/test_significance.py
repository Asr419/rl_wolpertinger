from scripts.simulation_imports import *
import os
import scipy.stats as stats

if __name__ == "__main__":
    ######SLATEQ########
    with open(
        "../../pomdp_saved_models/test_serving_observed_topic_slateq_140_05-01_19-39-45/logs_dict.pickle",
        "rb",
    ) as f:
        slate1 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_serving_observed_topic_slateq_184_05-01_19-42-21/logs_dict.pickle",
        "rb",
    ) as f:
        slate2 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_serving_observed_topic_slateq_30_05-01_19-42-56/logs_dict.pickle",
        "rb",
    ) as f:
        slate3 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_serving_observed_topic_slateq_4_05-01_19-44-46/logs_dict.pickle",
        "rb",
    ) as f:
        slate4 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_serving_observed_topic_slateq_62_05-01_19-43-35/logs_dict.pickle",
        "rb",
    ) as f:
        slate5 = pickle.load(f)

    ######WA_10##########
    with open(
        "../../pomdp_saved_models/test_wa_10_serving_observed_topic_slateq_140_05-01_19-50-00/logs_dict.pickle",
        "rb",
    ) as f:
        wa_10_1 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_10_serving_observed_topic_slateq_184_05-01_19-51-15/logs_dict.pickle",
        "rb",
    ) as f:
        wa_10_2 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_10_serving_observed_topic_slateq_30_05-01_19-51-46/logs_dict.pickle",
        "rb",
    ) as f:
        wa_10_3 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_10_serving_observed_topic_slateq_4_05-01_19-54-41/logs_dict.pickle",
        "rb",
    ) as f:
        wa_10_4 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_10_serving_observed_topic_slateq_62_05-01_19-52-19/logs_dict.pickle",
        "rb",
    ) as f:
        wa_10_5 = pickle.load(f)

    #######WA_20#######
    with open(
        "../../pomdp_saved_models/test_wa_20_serving_observed_topic_slateq_140_05-01_19-45-34/logs_dict.pickle",
        "rb",
    ) as f:
        wa_20_1 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_20_serving_observed_topic_slateq_184_05-01_19-46-06/logs_dict.pickle",
        "rb",
    ) as f:
        wa_20_2 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_20_serving_observed_topic_slateq_30_05-01_19-46-32/logs_dict.pickle",
        "rb",
    ) as f:
        wa_20_3 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_20_serving_observed_topic_slateq_4_05-01_19-48-29/logs_dict.pickle",
        "rb",
    ) as f:
        wa_20_4 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_20_serving_observed_topic_slateq_62_05-01_19-47-10/logs_dict.pickle",
        "rb",
    ) as f:
        wa_20_5 = pickle.load(f)

    ####WA_5####
    with open(
        "../../pomdp_saved_models/test_wa_5_serving_observed_topic_slateq_140_05-01_19-56-21/logs_dict.pickle",
        "rb",
    ) as f:
        wa_5_1 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_5_serving_observed_topic_slateq_184_05-01_19-56-55/logs_dict.pickle",
        "rb",
    ) as f:
        wa_5_2 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_5_serving_observed_topic_slateq_30_05-01_19-57-24/logs_dict.pickle",
        "rb",
    ) as f:
        wa_5_3 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_5_serving_observed_topic_slateq_4_05-01_19-58-23/logs_dict.pickle",
        "rb",
    ) as f:
        wa_5_4 = pickle.load(f)
    with open(
        "../../pomdp_saved_models/test_wa_5_serving_observed_topic_slateq_62_05-01_19-57-53/logs_dict.pickle",
        "rb",
    ) as f:
        wa_5_5 = pickle.load(f)
    slate1 = torch.mean(torch.tensor(slate1["ep_cum_reward"]))
    slate2 = torch.mean(torch.tensor(slate2["ep_cum_reward"]))
    slate3 = torch.mean(torch.tensor(slate3["ep_cum_reward"]))
    slate4 = torch.mean(torch.tensor(slate4["ep_cum_reward"]))
    slate5 = torch.mean(torch.tensor(slate5["ep_cum_reward"]))

    wa_10_1 = torch.mean(torch.tensor(wa_10_1["ep_cum_reward"]))
    wa_10_2 = torch.mean(torch.tensor(wa_10_2["ep_cum_reward"]))
    wa_10_3 = torch.mean(torch.tensor(wa_10_3["ep_cum_reward"]))
    wa_10_4 = torch.mean(torch.tensor(wa_10_4["ep_cum_reward"]))
    wa_10_5 = torch.mean(torch.tensor(wa_10_5["ep_cum_reward"]))

    wa_20_1 = torch.mean(torch.tensor(wa_20_1["ep_cum_reward"]))
    wa_20_2 = torch.mean(torch.tensor(wa_20_2["ep_cum_reward"]))
    wa_20_3 = torch.mean(torch.tensor(wa_20_3["ep_cum_reward"]))
    wa_20_4 = torch.mean(torch.tensor(wa_20_4["ep_cum_reward"]))
    wa_20_5 = torch.mean(torch.tensor(wa_20_5["ep_cum_reward"]))

    wa_5_1 = torch.mean(torch.tensor(wa_5_1["ep_cum_reward"]))
    wa_5_2 = torch.mean(torch.tensor(wa_5_2["ep_cum_reward"]))
    wa_5_3 = torch.mean(torch.tensor(wa_5_3["ep_cum_reward"]))
    wa_5_4 = torch.mean(torch.tensor(wa_5_4["ep_cum_reward"]))
    wa_5_5 = torch.mean(torch.tensor(wa_5_5["ep_cum_reward"]))

    slate = [slate1, slate2, slate3, slate4, slate5]
    wa_10 = [wa_10_1, wa_10_2, wa_10_3, wa_10_4, wa_10_5]
    wa_20 = [wa_20_1, wa_20_2, wa_20_3, wa_20_4, wa_20_5]
    wa_5 = [wa_5_1, wa_5_2, wa_5_3, wa_5_4, wa_5_5]
    print(torch.mean(torch.tensor(slate)))
    print(torch.mean(torch.tensor(wa_20)))
    print(torch.mean(torch.tensor(wa_10)))
    print(torch.mean(torch.tensor(wa_5)))

    # print(stats.shapiro(slate))
    # print(stats.shapiro(wa_10))
    # print(stats.shapiro(wa_20))
    # print(stats.shapiro(wa_5))

    # print("wolpertinger_20")
    # print(stats.ttest_rel(slate, wa_20))
    # print(stats.wilcoxon(slate, wa_20))
    # print("wolpertinger_10")
    # print(stats.ttest_rel(slate, wa_10))
    # print(stats.wilcoxon(slate, wa_10))
    # print("wolpertinger_5")
    # print(stats.ttest_rel(slate, wa_5))
    # print(stats.wilcoxon(slate, wa_5))
