import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats
import random
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
from user_sampler import LTSStaticUserSampler
from user_sampler import LTSUserState
from document_sampler import LTSDocumentSampler


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
    self._generate_response(slate_documents[selected_index], responses[selected_index])
    return responses


def generate_response(self, doc, response):
    response.clicked = True
    # linear interpolation between choc and kale.
    if doc.age > 40:
        engagement_loc = 1 / abs(
            (
                (doc.acousticness - self._user_state.acousticness)
                + (doc.liveness - self._user_state.liveness)
            )
        )
        engagement_loc *= self._user_state.satisfaction
        engagement_scale = doc.acousticness * (self._user_state.mood + 1) + (
            (1 - doc.liveness) * (self._user_state.mood + 1)
        )
    else:
        engagement_loc = 1 / abs(
            (doc.danceability - self._user_state.danceability)
            + (doc.energy - self._user_state.energy)
            + (doc.valence - self._user_state.valence)
        )
        engagement_loc *= self._user_state.satisfaction
        engagement_scale = doc.danceabilty * (self._user_state.mood + 1) + (
            (1 - doc.energy) * (self._user_state.mood + 1)
        )
    log_engagement = np.random.normal(loc=engagement_loc, scale=engagement_scale)
    response.engagement = np.exp(log_engagement)


def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
        if response.clicked:
            mood = np.random.normal(scale=self._user_state.mood)
            net_genre_exposure = (
                self._user_state.valence * self._user_state.danceability
                - 2.0 * (doc.danceability - 0.5)
                + mood
            )
            self._user_state.net_genre_exposure = net_genre_exposure
            satisfaction = 1 / (
                1.0 + np.exp(-self._user_state.mood * net_genre_exposure)
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

    slate_size = 3
    num_candidates = 10
    ltsenv = environment.Environment(
        LTSUserModel(slate_size),
        LTSDocumentSampler(),
        num_candidates,
        slate_size,
        resample_documents=True,
    )
