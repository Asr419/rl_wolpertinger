import abc
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt
from recsim.choice_model import AbstractChoiceModel


class AbstractUserFeaturesGenerator(metaclass=abc.ABCMeta):
    # class modeling generators for user state
    def __init__(self, num_features: int) -> None:
        self.num_features = num_features
        pass

    @abc.abstractmethod
    def __call__(*args: Any, **kwds: Any) -> npt.NDArray[np.float64]:
        """Generate a user state

        Returns:
            npt.NDArray[np.float64]: user state
        """
        pass


class NormalUserFeaturesGenerator(AbstractUserFeaturesGenerator):
    """Normal distribution user state generator"""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, num_features: int) -> npt.NDArray[np.float64]:
        return np.random.normal(self.mean, self.std, num_features)
