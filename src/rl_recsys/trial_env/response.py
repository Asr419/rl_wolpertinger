import random

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from recsim import document, user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment, recsim_gym
from scipy import stats

from rl_recsys.trial_env.document_sampler import LTSDocumentSampler
from rl_recsys.trial_env.user_sampler import LTSStaticUserSampler, LTSUserState


class LTSResponse(user.AbstractResponse):
    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, clicked=False, engagement=0.0):
        self.clicked = clicked
        self.engagement = engagement

    def create_observation(self):
        return {"click": int(self.clicked), "engagement": np.array(self.engagement)}

    @classmethod
    def response_space(cls):
        # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
        return spaces.Dict(
            {
                "click": spaces.Discrete(2),
                "engagement": spaces.Box(
                    low=0.0,
                    high=cls.MAX_ENGAGEMENT_MAGNITUDE,
                    shape=tuple(),
                    dtype=np.float32,
                ),
            }
        )


def user_init(self, slate_size, seed=0):
    super(LTSUserModel, self).__init__(
        LTSResponse, LTSStaticUserSampler(LTSUserState, seed=seed), slate_size
    )
    self.choice_model = MultinomialLogitChoiceModel({})


def simulate_response(self, slate_documents):
    # List of empty responses
    responses = [self._response_model_ctor() for _ in slate_documents]
    # Get click from of choice model.
    self.choice_model.score_documents(
        self._user_state, [doc.create_observation() for doc in slate_documents]
    )
    scores = self.choice_model.scores
    selected_index = self.choice_model.choose_item()
    # Populate clicked item.
    print(slate_documents[selected_index])
    self._generate_response(slate_documents[selected_index], responses[selected_index])
    return responses


#   def simulate_response(self, slate_documents):
#     # List of empty responses
#     responses = [self._response_model_ctor() for _ in slate_documents]
#     # Get click from of choice model.
#     self.choice_model.score_documents(
#         self._user_state, [doc.create_observation() for doc in slate_documents]
#     )
#     scores = self.choice_model.scores
#     selected_index = self.choice_model.choose_item()
#     session_engagement = 0
#     session_items = []
#     individual_reward = []
#     # Populate clicked item.
#     for i in range(0, len(slate_documents)):
#         k = self._generate_response(slate_documents[i], responses[i])
#         p = k.index.tolist()
#         session_items.append(p[0])
#         session_engagement += k.values
#         m = k.values.tolist()
#         individual_reward.append(m[0])

#     # self._generate_response(slate_documents[selected_index],
#     #                         responses[selected_index])
#     user_information = [
#         self._user_state.age,
#         self._user_state.gender,
#         self._user_state.acousticness,
#         self._user_state.liveness,
#         self._user_state.danceability,
#         self._user_state.valence,
#         self._user_state.label,
#     ]
#     print(user_information)
#     print(session_items)
#     print(individual_reward)
#     print(session_engagement)
#     return responses


def generate_response(self, doc, response):
    response.clicked = True
    # linear interpolation between choc and kale.
    if self._user_state.age > 40:
        engagement = (
            (doc.acousticness - self._user_state.acousticness)
            + (doc.liveness - self._user_state.liveness)
        ) + (doc.label - self._user_state.label)
        engagement_loc = 1 / abs(
            (
                (doc.acousticness - self._user_state.acousticness)
                + (doc.liveness - self._user_state.liveness)
            )
        )
        engagement_loc *= self._user_state.satisfaction
        engagement_scale = doc.acousticness * (self._user_state.label + 1) + (
            (1 - doc.liveness) * (self._user_state.label + 1)
        )
    else:
        engagement = (
            (doc.danceability - self._user_state.danceability)
            + (doc.valence - self._user_state.valence)
            + (doc.label - self._user_state.label)
        )
        engagement_loc = 1 / abs(
            (doc.danceability - self._user_state.danceability)
            + (doc.valence - self._user_state.valence)
        )
        engagement_loc *= self._user_state.satisfaction
        engagement_scale = doc.danceability * (self._user_state.label + 1) + (
            (1 - doc.valence) * (self._user_state.label + 1)
        )
    log_engagement = np.random.normal(loc=engagement_loc, scale=engagement_scale)

    response.engagement = np.ceil(engagement)
    return engagement


def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
        if response.clicked:
            label = np.random.normal(scale=self._user_state.label)
            net_genre_exposure = (
                self._user_state.valence * self._user_state.danceability
                - 2.0 * (doc.danceability - 0.5)
                + label
            )
            self._user_state.net_genre_exposure = net_genre_exposure
            satisfaction = 1 / (
                1.0 + np.exp(-self._user_state.label * net_genre_exposure)
            )
            self._user_state.satisfaction = satisfaction
            self._user_state.time_budget -= 1
            return


def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return self._user_state.time_budget <= 0


if __name__ == "__main__":
    LTSUserModel = type(
        "LTSUserModel",
        (user.AbstractUserModel,),
        {
            "__init__": user_init,
            "is_terminal": is_terminal,
            "update_state": update_state,
            "simulate_response": simulate_response,
            "_generate_response": generate_response,
        },
    )

    slate_size = 10
    num_candidates = 10
    ltsenv = environment.Environment(
        LTSUserModel(slate_size),
        LTSDocumentSampler(),
        num_candidates,
        slate_size,
        resample_documents=True,
    )

    def clicked_engagement_reward(responses):
        reward = 0.0
        for response in responses:
            # if response.clicked:
            reward += response.engagement
        return reward

    def slate_score(env):
        observation_0 = env.reset()

        doc_strings = [
            "music_id " + key + " index " + str(value)
            for key, value in observation_0["doc"].items()
        ]
        recommendation_slate_0 = [i for i in range(0, 10)]
        for i in range(0, 10):
            observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)

    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)

    observation_0 = lts_gym_env.reset()
    print("Observation 0")
    print("Available documents")
    doc_strings = [
        "music_id " + key + " index " + str(value)
        for key, value in observation_0["doc"].items()
    ]
    print("\n".join(doc_strings))
    print("Noisy user state observation")
    print(observation_0["user"])
    # Agent recommends the first three documents.
    recommendation_slate_0 = [0, 1, 2]
    observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
    print("Observation 1")
    print("Available documents")
    doc_strings = [
        "music_id " + key + " index " + str(value)
        for key, value in observation_1["doc"].items()
    ]
    print("\n".join(doc_strings))
    rsp_strings = [str(response) for response in observation_1["response"]]
    print("User responses to documents in the slate")
    print("\n".join(rsp_strings))
    print("Noisy user state observation")
    print(observation_1["user"])