import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt


class AbstractUserModel:
    def __init__(
        self,
        user_features: npt.NDArray[np.float64],
        user_state: npt.NDArray[np.float64],
        user_choice_model: user_choice_model,
        songs_per_sess: int = 50,
        avg_song_duration: float = 207467.0,
    ) -> None:
        # init budget
        self.budget = songs_per_sess * avg_song_duration
        self.p_u = user_state

        # call generate state
        self.user_state = self.user_state.generate_state()
        self.user_choice_model = user_choice_model
        self.user_features = user_features


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


# user_state_gen = TypeVar("user_state_gen", bound=AbstractUserStateGenerator)
# user_choice_model = TypeVar("user_choice_model", bound=AbstractChoiceModel)
