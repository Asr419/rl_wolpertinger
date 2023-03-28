import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt


class AbstractUserState(metaclass=abc.ABCMeta):
    # hidden state of the user
    def __init__(self, **kwds: Any) -> None:
        self.user_state = self.generate_state(**kwds)

    @abc.abstractmethod
    def generate_state(self, **kwds: Any) -> npt.NDArray[np.float64]:
        """Generate the user hidden state"""
        pass

    @abc.abstractmethod
    def update_state(self, **kwds: Any) -> None:
        """update the user hidden state"""
        pass


class AlphaIntentUserState(AbstractUserState):
    def __init__(self, user_features: npt.NDArray[np.float64]) -> None:
        # called p_u in the paper
        self.user_features = user_features
        self.tgt_feature_idx = np.random.randint(0, len(user_features))
        # called p_uh in the paper
        self.user_state = self.generate_state(self.user_features)

    def generate_state(
        self, user_features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        user_state = user_features.copy()
        # sample alpha from a uniform distribution
        alpha = np.random.uniform(0, 1)
        # alpha = 1
        inv_alpha = 1 - alpha

        # creating tgt feature mask and inverse mask
        feat_mask = np.zeros(len(user_state))
        inv_feat_mask = np.ones(len(user_state))
        # select target feature randomly
        feat_mask[self.tgt_feature_idx] = 1
        inv_feat_mask[self.tgt_feature_idx] = 0

        user_state[feat_mask == 1] = alpha * user_state[feat_mask == 1]
        user_state[inv_feat_mask == 1] = (
            inv_alpha / (len(user_state) - 1) * user_state[inv_feat_mask == 1]
        )

        return user_state

    def update_state(self, **kwds: Any) -> None:
        # no update for the user state
        # TODO: add update for the user state
        pass
