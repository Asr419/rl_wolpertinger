import abc
from typing import TypeVar

import torch

from rl_recsys.belief_modeling.belief_model import NNBeliefModel
from rl_recsys.agent_modeling.slate_generator import Topk_slate

torch_model = TypeVar("torch_model", bound=torch.nn.Module)


class AbstractSlateAgent(metaclass=abc.ABCMeta):
    # model an abstract agent recommending slates of documents
    pass

    def __init__(self, slate_gen_func, state, candidate_docs) -> None:
        self.slate_gen_func = slate_gen_func
        self.state = state
        self.candidate_docs = candidate_docs
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


class Rl_agent(AbstractSlateAgent):
    def __init__(self, slate_gen_func, state, candidate_docs) -> None:
        self.state = state
        self.slate_gen_func = slate_gen_func
        self.candidate_docs = candidate_docs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def get_action(slate_gen_func, state, candidate_docs):
        return slate_gen_func(state, candidate_docs)
