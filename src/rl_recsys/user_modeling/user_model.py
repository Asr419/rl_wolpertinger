import abc
from dataclasses import dataclass
from typing import Any, Callable, List, Type, TypeVar

import numpy as np
import numpy.typing as npt

from rl_recsys.user_modeling.choice_model import AbstractChoiceModel
from rl_recsys.user_modeling.features_gen import AbstractFeaturesGenerator
from rl_recsys.user_modeling.response_model import AbstractResponseModel
from rl_recsys.user_modeling.user_state import AbstractUserState

user_state_model_type = TypeVar("user_state_model_type", bound=AbstractUserState)
user_choice_model_type = TypeVar("user_choice_model_type", bound=AbstractChoiceModel)
user_response_model_type = TypeVar(
    "user_response_model_type", bound=AbstractResponseModel
)
feature_gen_type = TypeVar("feature_gen_type", bound=AbstractFeaturesGenerator)


class UserModel:
    def __init__(
        self,
        user_features: npt.NDArray[np.float64],
        user_state_model: user_state_model_type,
        user_choice_model: user_choice_model_type,
        user_response_model: user_response_model_type,
        songs_per_sess: int = 50,
        avg_song_duration: float = 207467.0,
    ) -> None:
        self.budget = songs_per_sess * avg_song_duration
        self.state_model = user_state_model
        self.choice_model = user_choice_model
        self.response_model = user_response_model
        self.features = user_features

        self.song_per_sess = songs_per_sess
        self.avg_song_duration = avg_song_duration

    def get_state(self):
        return self.state_model.user_state

    def is_terminal(self) -> bool:
        return self.budget <= 0

    def update_budget(self, selected_doc_duration: float) -> None:
        self.budget -= selected_doc_duration

    def update_budget_avg(self) -> None:
        self.budget -= self.avg_song_duration


class UserSampler:
    # has to call user features generator to initialize a user
    def __init__(
        self,
        user_feature_gen: feature_gen_type,
        state_model_cls: type[user_state_model_type],
        choice_model_cls: type[user_choice_model_type],
        response_model_cls: type[user_response_model_type],
        num_user_features: int = 14,
    ) -> None:
        self.state_model_cls = state_model_cls
        self.choice_model_cls = choice_model_cls
        self.response_model_cls = response_model_cls
        self.feature_gen = user_feature_gen
        self.num_user_features = num_user_features

        self.users: List[UserModel] = []

    def _generate_user(self) -> UserModel:
        # generate a user
        user_features = self.feature_gen(num_features=self.num_user_features)

        # initialize models
        state_model = self.state_model_cls(user_features=user_features)
        choice_model = self.choice_model_cls(user_state=state_model.user_state)
        response_model = self.response_model_cls(user_state=state_model.user_state)

        user = UserModel(
            user_features=user_features,
            user_state_model=state_model,
            user_choice_model=choice_model,
            user_response_model=response_model,
        )

        return user

    def generate_users(self, num_users: int = 100) -> List[UserModel]:
        self.users = [self._generate_user() for _ in range(num_users)]
        return self.users

    def sample_user(self) -> UserModel:
        assert (
            len(self.users) > 0
        ), "No users generated yet. call generate_user_batch() first.)"
        i = np.random.randint(0, len(self.users))
        return self.users[i]
