import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from rl_recsys.user_modeling.features_gen import AbstractFeaturesGenerator

feature_gen_type = TypeVar("feature_gen_type", bound=AbstractFeaturesGenerator)


class AbstractUserState(nn.Module, metaclass=abc.ABCMeta):
    # hidden state of the user
    def __init__(self, state_update_rate: float, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.state_update_rate = state_update_rate

    @abc.abstractmethod
    def _generate_state(self, **kwds: Any) -> torch.Tensor:
        """Generate the user hidden state"""
        pass

    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        w = self.state_update_rate
        self.user_state = w * self.user_state + (1 - w) * selected_doc_feature


class AlphaIntentUserState(AbstractUserState):
    def __init__(
        self,
        user_features: torch.Tensor,
        intent_gen: feature_gen_type,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        num_user_features = len(user_features)
        self.intent_gen = intent_gen

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # self.alpha = np.random.uniform(self.alpha_min, self.alpha_max, 1).astype(
        #     np.float32
        # )
        self.alpha = 0.2

        self.intent = self.intent_gen(num_user_features)

        user_state = self._generate_state(user_features=user_features)
        self.register_buffer("user_state", user_state)
        # used to reset the intent to the initial create one at the end of an episode
        self.register_buffer("user_state_init", user_state)

    def _generate_state(self, user_features: torch.Tensor) -> torch.Tensor:
        user_state = user_features * (1 - self.alpha) + self.intent * self.alpha
        # print(
        #     f"alpha:{self.alpha}\nuser_features: {user_features}\n user_intent: {self.intent}\nuser_state: {user_state}\n\n"
        # )
        return user_state

    def reset_state(self) -> None:
        self.user_state = self.user_state_init


# class AlphaIntentUserStateOld(AbstractUserState):
#     def __init__(self, user_features: torch.Tensor, **kwds: Any) -> None:
#         super().__init__(**kwds)
#         self.user_features = user_features

#         # select a target feature for the user among the positve ones
#         tgt_feature_idx = None
#         tgt_feat_val = -1
#         while tgt_feat_val < 0:
#             tgt_feature_idx = torch.randint(0, len(user_features), size=(1,))
#             tgt_feat_val = user_features[tgt_feature_idx]
#         self.tgt_feature_idx = tgt_feature_idx

#         user_state = self._generate_state(self.user_features)
#         self.register_buffer("user_state", user_state)
#         # used to reset the intent to the initial create one at the end of an episode
#         self.register_buffer("user_state_init", user_state)

#     def _generate_state(self, user_features: npt.NDArray[np.float_]) -> torch.Tensor:
#         user_state = user_features.cpu().numpy().copy()  # type: ignore

#         tgt_feature_idx = None
#         tgt_feat_val = -1
#         while tgt_feat_val < 0:
#             tgt_feature_idx = np.random.randint(0, len(user_features), size=(1,))
#             tgt_feat_val = user_features[tgt_feature_idx]

#         # sample alpha from a uniform distribution
#         alpha = np.random.uniform()
#         # print(alpha)
#         # print(tgt_feature_idx)

#         pos_sum = np.sum(user_state[user_state > 0])
#         pos_val_count = len(user_state[user_state > 0])

#         user_state[tgt_feature_idx] = user_state[tgt_feature_idx] + (2 * alpha)
#         user_state[user_state > 0] -= alpha / pos_val_count

#         if user_state[tgt_feature_idx] > 1:
#             user_state[user_state > 0] /= user_state[tgt_feature_idx]

#         user_state = torch.Tensor(user_state)
#         # print(user_state)
#         # print(user_features)
#         return user_state

#     def reset_state(self) -> None:
#         self.user_state = self.user_state_init
