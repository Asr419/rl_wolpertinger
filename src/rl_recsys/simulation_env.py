from typing import Tuple

import gym

from rl_recsys.user_modeling.user_model import UserModel


class DynamicUserEnv(gym.Env):
    def __init__(self, user_sampler, doc_catalogue, rec_model) -> None:
        self.user_sampler = user_sampler
        self.doc_catalogue = doc_catalogue
        self.rec_model = rec_model

    def step(self, action):
        pass

    def reset(self) -> Tuple[UserModel, list[int]]:
        """
        Reset the environment's state.

        A new user is sampled and the candidate documents are retrieved.
        Returns:
            Tuple[UserModel, list[int]]: the user and the candidate documents
        """
        user = self.user_sampler.sample_user()
        # retrieve documents based on p_u
        candidate_docs = self.rec_model.recommend(user.features)
        return user, candidate_docs

    def render(self):
        raise NotImplementedError()
