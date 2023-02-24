import abc
from dataclasses import dataclass
from typing import Any, Callable, List, Type, TypeVar

import numpy as np
import numpy.typing as npt

from rl_recsys.user_modeling.choice_model import AbstractChoiceModel
from rl_recsys.user_modeling.response_model import AbstractResponseModel
from rl_recsys.user_modeling.user_features_gen import AbstractUserFeaturesGenerator
from rl_recsys.user_modeling.user_state import AbstractUserState

user_state_model_type = TypeVar("user_state_model_type", bound=AbstractUserState)
user_choice_model_type = TypeVar("user_choice_model_type", bound=AbstractChoiceModel)
user_response_model_type = TypeVar(
    "user_response_model_type", bound=AbstractResponseModel
)
user_feature_gen_type = TypeVar("feature_gen_type", bound=AbstractUserFeaturesGenerator)


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

    def is_terminal(self) -> bool:
        return self.budget <= 0

    def update_budget(self, selected_doc_duration: float) -> None:
        self.budget -= selected_doc_duration


class UserSampler:
    # has to call user features generator to initialize a user
    def __init__(
        self,
        user_feature_gen: user_feature_gen_type,
        state_model_cls: type[user_state_model_type],
        choice_model_cls: type[user_choice_model_type],
        response_model_cls: type[user_response_model_type],
    ) -> None:
        self.state_model_cls = state_model_cls
        self.choice_model_cls = choice_model_cls
        self.response_model_cls = response_model_cls
        self.feature_gen = user_feature_gen

        self.users: List[UserModel]

    def generate_user(self, num_users: int = 100) -> UserModel:
        # generate a user
        user_features = self.feature_gen()

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

    def generate_user_batch(self, num_users: int = 100) -> List[UserModel]:
        self.users = [self.generate_user() for _ in range(num_users)]
        return self.users
