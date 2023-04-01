import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch


class AbstractUserState(metaclass=abc.ABCMeta):
    # hidden state of the user
    def __init__(self, **kwds: Any) -> None:
        self.user_state = self.generate_state(**kwds)

    @abc.abstractmethod
    def generate_state(self, **kwds: Any) -> torch.Tensor:
        """Generate the user hidden state"""
        pass

    def update_state(self, selected_doc_feature: torch.Tensor) -> None:
        self.user_state = torch.mean(
            torch.stack((selected_doc_feature, self.user_state)), dim=0
        )


class AlphaIntentUserState(AbstractUserState):
    def __init__(self, user_features: torch.Tensor) -> None:
        # called p_u in the paper
        self.user_features = user_features
        self.tgt_feature_idx = torch.randint(0, len(user_features), size=(1,))
        # called p_uh in the paper
        self.user_state = self.generate_state(self.user_features)

    def generate_state(self, user_features: npt.NDArray[np.float_]) -> torch.Tensor:
        user_state = torch.Tensor(user_features).clone()
        # sample alpha from a uniform distribution
        alpha = torch.rand(1)
        # alpha = 1
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
