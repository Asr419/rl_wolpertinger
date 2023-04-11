import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class AbstractUserState(nn.Module, metaclass=abc.ABCMeta):
    # hidden state of the user
    def __init__(self, state_update_rate: float, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.state_update_rate = state_update_rate

    @abc.abstractmethod
    def generate_state(self, **kwds: Any) -> torch.Tensor:
        """Generate the user hidden state"""
        pass

    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        # generate a random integer between 0 and 1
        random = np.random.randint(0, 1)
        if random > 0.7:
            # 10% chance of boredom
            w = self.state_update_rate
            self.user_state = w * self.user_state - (1 - w) * selected_doc_feature
        else:
            w = self.state_update_rate
            self.user_state = w * self.user_state + (1 - w) * selected_doc_feature


class AlphaIntentUserState(AbstractUserState):
    def __init__(self, user_features: torch.Tensor, **kwds: Any) -> None:
        super().__init__(**kwds)
        # called p_u in the paper
        self.user_features = user_features

        # assure tgt feature is positive
        tgt_feature_idx = None
        tgt_feat_val = -1
        while tgt_feat_val < 0:
            tgt_feature_idx = torch.randint(0, len(user_features), size=(1,))
            tgt_feat_val = user_features[tgt_feature_idx]

        self.tgt_feature_idx = tgt_feature_idx
        # called p_uh in the paper
        user_state = self.generate_state(self.user_features)
        self.user_state_init = user_state.clone()
        self.register_buffer("user_state", user_state)

    def generate_state(self, user_features: torch.Tensor) -> torch.Tensor:
        user_state = torch.Tensor(user_features).clone()
        # sample alpha from a uniform distribution
        # alpha = torch.rand(1)
        alpha = 0.8
        alpha = 0.8 * alpha + 0.2  # alpha between 0.2 and 1

        inv_alpha = 1 - alpha

        # creating tgt feature mask and inverse mask
        feat_mask = torch.zeros(len(user_state))
        inv_feat_mask = torch.ones(len(user_state))
        # select target feature randomly
        feat_mask[self.tgt_feature_idx] = 1
        inv_feat_mask[self.tgt_feature_idx] = 0

        user_state[feat_mask == 1] = alpha * user_state[feat_mask == 1]
        user_state[inv_feat_mask == 1] = (
            inv_alpha / (len(user_state) - 1) * user_state[inv_feat_mask == 1]
        )

        return user_state

    def reset_state(self) -> None:
        self.user_state = self.user_state_init
