import random as rand
from math import sqrt
import numpy as np
import pandas as pd

from rl_recsys.user_modeling.choice_model import DeterministicChoicheModel
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.belief_modeling import belief_model
from rl_recsys.user_modeling.user_model import UserModel
from rl_recsys.user_modeling.user_model import UserSampler

from gym import Env
from gym.spaces import Discrete, Box
from numpy import int64


class MusicGym(Env):
    def __init__(self, user_sampler, doc_catalogue, rec_model) -> None:
        self.user_sampler = user_sampler
        self.doc_catalogue = doc_catalogue
        self.rec_model = rec_model

    def step(self, action):
        observation = DeterministicChoicheModel.choose_document(self.state, action)
        self.next_belief_state = belief_model(self.belief_state, action)
        reward = np.dot(self.belief_state, observation)
        budget = UserModel.update_budget(observation)
        if budget <= 0:
            done = True
        else:
            done = False
        info = {}
        return self.belief_state, reward, self.next_belief_state, done

    def reset(self):
        user = self.UserSampler.sample_user()
        candidate_docs = self.rec_model.recommend(user.features)
        return user, candidate_docs

    def render(self):
        raise NotImplementedError()
