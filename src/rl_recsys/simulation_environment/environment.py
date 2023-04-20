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
        self.candidate_docs: torch.Tensor

    def step(self, slate: npt.NDArray[np.int_], candidate_docs):
        # action: is the slate created by the agent
        # observation: is the selected document in the slate

        # retrieving fetaures of the slate documents
        
        
        slate_doc_ids = candidate_docs[slate]
        
        doc_features = torch.Tensor(
            self.doc_catalogue.get_docs_features(slate_doc_ids)
        ).to(device=self.device)

        # select from the slate on item following the user choice model
        self.curr_user.choice_model.score_documents(
            self.curr_user.get_state(), doc_features
        )

        selected_doc_idx = self.curr_user.choice_model.choose_document()

        # ???
        doc_id = slate[selected_doc_idx]
        

        # check if user has selected a document
        selected_doc_feature = doc_features[selected_doc_idx, :]
        response = self.curr_user.response_model.generate_response(
            self.curr_user.get_state(), selected_doc_feature
        )

        # if selected_doc_idx == self.curr_user.choice_model.no_selection_token:
        #     response = self.curr_user.response_model.generate_null_response().to(
        #         device=self.device
        #     )
        #     selected_doc_feature = torch.zeros(14).to(self.device)
        # else:
        #     selected_doc_feature = doc_features[selected_doc_idx, :]
        #     response = self.curr_user.response_model.generate_response(
        #         self.curr_user.get_state(), selected_doc_feature
        #     )
        # update user state
        self.curr_user.state_model.update_state(
            selected_doc_feature=selected_doc_feature
        )
        if selected_doc_idx == self.curr_user.choice_model.no_selection_token:
            print(self.curr_user.get_state())

        # self.curr_user.update_budget(response)
        self.curr_user.update_budget_avg()

        is_terminal = self.curr_user.is_terminal()
        info = {}
        return selected_doc_feature, response, is_terminal, False, info

    def reset(self) -> None:
        # initialize an episode by setting the user and the candidate documents
        user = self.user_sampler.sample_user()
        self.curr_user = user
        # initialize user budget
        user.budget = user.init_budget()
        # reinitialize the user state
        self.curr_user.state_model.reset_state()
        # initialize user hidden state
        # retrieve candidate documents
        candidate_docs = self.rec_model.recommend_dot(user.features.to("cpu"), self.k)
        # candidate_docs = self.rec_model.recommend_random(k=self.k)
        self.candidate_docs = torch.Tensor(candidate_docs).to(device=self.device)
        
        # candidate_docs_repr = torch.Tensor(
        #     self.doc_catalogue.get_docs_features(candidate_docs)
        # )
        # cos_sim = torch.nn.functional.cosine_similarity(
        #     self.curr_user.get_state(), candidate_docs_repr, dim=1
        # )
        # self.satisfaction_score = (cos_sim.mean() + 1) / 2

    def render(self):
        raise NotImplementedError()

    def get_candidate_docs(self) -> torch.Tensor:
        return self.candidate_docs
