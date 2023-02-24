import abc
from typing import Any, Callable, Type, TypeVar

import numpy as np
import numpy.typing as npt

from rl_recsys.user_modeling.choice_model import AbstractChoiceModel
from rl_recsys.user_modeling.response_model import AbstractResponseModel
from rl_recsys.user_modeling.user_state import AbstractUserState

user_state_model_type = TypeVar("user_state_model_type", bound=AbstractUserState)
user_choice_model_type = TypeVar("user_choice_model_type", bound=AbstractChoiceModel)
user_response_model_type = TypeVar(
    "user_response_model_type", bound=AbstractResponseModel
)


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


# TODO: user sampler, user choice model


class UserSampler:
    # has to call user features generator to initialize a user
    def generate_user(
        self, user_features_gen: user_features_gen, user_state_gen: user_state_gen
    ) -> UserModel:
        user_features = user_features_gen()
        user_state = user_state_gen()
        user_choice_model = self.user_choice_model
        return UserModel(user_features, user_state, user_choice_model)

    def generate_batch_users(
        self, user_features_gen: user_features_gen, user_state_gen: user_state_gen
    ) -> List[UserModel]:
        return [
            self.generate_user(user_features_gen, user_state_gen)
            for _ in range(self.batch_size)
        ]
