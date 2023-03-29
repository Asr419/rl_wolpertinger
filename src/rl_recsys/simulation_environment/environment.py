import random as rand
from math import sqrt

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from rl_recsys.belief_modeling import belief_model
from rl_recsys.user_modeling.user_model import UserModel, UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState


class MusicGym(gym.Env):
    def __init__(
        self,
        user_sampler,
        doc_catalogue,
        rec_model,
        k: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.user_sampler = user_sampler
        self.doc_catalogue = doc_catalogue
        self.rec_model = rec_model
        self.device = device
        # number of candidates items
        self.k = k

        # initialized by reset
        self.curr_user: UserModel
        self.candidate_docs: list[int]

    def step(self, slate: npt.NDArray[np.int_], belief_state: torch.Tensor):
        # action: is the slate created by the agent
        # observation: is the selected document in the slate

        # retrieving fetaures of the slate documents
        doc_features = torch.Tensor(self.doc_catalogue.get_docs_features(slate)).to(
            device=self.device
        )
        # p_uh = torch.Tensor(self.curr_user.get_state()).to(self.device)
        # select from the slate on item following the user choice model
        self.curr_user.choice_model.score_documents(self.p_uh, doc_features)

        selected_doc_idx = self.curr_user.choice_model.choose_document()

        # ???
        doc_id = slate[selected_doc_idx]

        # get feature of the selected document
        if selected_doc_idx > 0:
            selected_doc_feature = doc_features[selected_doc_idx, :]
        else:
            selected_doc_feature = torch.zeros(14)

        self.p_uh = torch.Tensor(self.curr_user.update_state(selected_doc_feature)).to(
            self.device
        )
        # compute the reward
        if torch.any(selected_doc_feature != 0):
            response = self.curr_user.response_model.generate_response(
                self.p_uh, selected_doc_feature
            )

        else:
            response = torch.tensor(-10.0)

        # response = self.curr_user.response_model.generate_response(
        #     belief_state, selected_doc_feature
        # )
        # if response >= 7:
        #     response = response
        # else:
        #     response = torch.tensor(-10.0)
        # update the budget
        # self.curr_user.update_budget_avg()
        self.curr_user.update_budget(response)

        is_terminal = self.curr_user.is_terminal()
        info = {}
        return selected_doc_feature, response, is_terminal, False, info

    def reset(self) -> None:
        # initialize an episode by setting the user and the candidate documents
        user = self.user_sampler.sample_user()
        self.curr_user = user
        user.budget = user.init_budget()
        self.p_uh = torch.Tensor(self.curr_user.get_state()).to(self.device)
        self.candidate_docs = self.rec_model.recommend_random(user.features, self.k)

    def render(self):
        raise NotImplementedError()

    def get_curr_state(self) -> npt.NDArray[np.float_]:
        return self.curr_user.get_state()

    def get_candidate_docs(self) -> npt.NDArray[np.int_]:
        return np.array(self.candidate_docs)
