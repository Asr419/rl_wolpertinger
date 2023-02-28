import abc
from typing import TypeVar

import torch

torch_model = TypeVar("torch_model", bound=torch.nn.Module)


class AbstractSlateAgent(metaclass=abc.ABCMeta):
    # model an abstract agent recommending slates of documents
    pass

    def __init__(
        self,
        slate_gen_func,
    ) -> None:
        self.slate_gen_func = slate_gen_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    @abc.abstractmethod
    def init_state(**kwargs) -> torch.Tensor:
        """Initialize the first estimated state of the agent"""
        pass


class AbstractBeliefAgent(metaclass=abc.ABCMeta):
    # model an abstract agent with a belief state
    def __init__(self, agent: torch_model, belief_model: torch_model) -> None:
        self.agent = agent
        self.belief_model = belief_model

    def update_belief(self, *args, **kwargs) -> torch.Tensor:
        """Update the belief state of the agent"""
        return self.belief_model(*args, **kwargs)
