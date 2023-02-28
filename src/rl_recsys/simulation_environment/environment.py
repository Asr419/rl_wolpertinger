import random as rand
from math import sqrt

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd

from rl_recsys.belief_modeling import belief_model
from rl_recsys.user_modeling.choice_model import DeterministicChoicheModel
from rl_recsys.user_modeling.user_model import UserModel, UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState


class MusicGym(gym.Env):
    def __init__(
        self,
        user_sampler,
        doc_catalogue,
        rec_model,
        k: int = 10,
    ) -> None:
        self.user_sampler = user_sampler
        self.doc_catalogue = doc_catalogue
        self.rec_model = rec_model
        # number of candidates items
        self.k = k

        # initialized by reset
        self.curr_user: UserModel
        self.candidate_docs: list[int]

    def step(self, slate: list[int], belief_state: npt.NDArray[np.float_]):
        # action: is the slate created by the agent
        # observation: is the selected document in the slate

        doc_features = self.doc_catalogue.get_docs_features(slate)
        selected_doc_idx = self.curr_user.choice_model.choose_document(
            self.curr_user.get_state(), doc_features
        )
        doc_id = slate[selected_doc_idx]
        selected_doc_feature = doc_features[selected_doc_idx, :]

        # belief update goes into the agent
        # self.next_belief_state = belief_model(self.belief_state, action)

        # compute the reward
        response = self.curr_user.response_model.generate_response(belief_state, doc_id)
        self.curr_user.update_budget(song_duration)

        terminated = self.curr_user.is_terminal()
        info = {}
        return selected_doc_feature, response, terminated, False, info

    def reset(self) -> None:
        # initialize an episode by setting the user and the candidate documents
        user = self.user_sampler.sample_user()
        self.curr_user = user
        self.candidate_docs = self.rec_model.recommend(user.features)

    def render(self):
        raise NotImplementedError()
