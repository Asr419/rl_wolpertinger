import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt


class AbstractUserState:
    # hidden state of the user
    @abc.abstractmethod
    def generate_state(self, **kwds: Any) -> npt.NDArray[np.float64]:
        """Generate the user hidden state"""
        pass

    @abc.abstractmethod
    def update_state(self, **kwds: Any) -> None:
        """update the user hidden state"""
        pass


class IntentUserState(AbstractUserState):
    def generate_state(
        self, user_features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # todo
        # select target feature randomly
        tgt_feature_idx = np.random.randint(0, len(user_features))
        pass
