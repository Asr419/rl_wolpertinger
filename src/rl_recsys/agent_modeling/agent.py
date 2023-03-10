import abc
from typing import TypeVar

import torch
import torch.nn as nn

from rl_recsys.agent_modeling.slate_generator import Topk_slate
from rl_recsys.belief_modeling.belief_model import NNBeliefModel

torch_model = TypeVar("torch_model", bound=torch.nn.Module)


class BeliefAgent(
    nn.Module,
):
    # model an abstract agent with a belief state
    def __init__(self, agent: torch_model, belief_model: torch_model) -> None:
        self.agent = agent
        self.belief_model = belief_model

    def update_belief(self, *args, **kwargs) -> torch.Tensor:
        """Update the belief state of the agent"""
        return self.belief_model(*args, **kwargs)


class AbstractSlateAgent(metaclass=abc.ABCMeta):
    # model an abstract agent recommending slates of documents
    def __init__(self, slate_gen_func) -> None:
        self.slate_gen_func = slate_gen_func

    def get_action(
        self, state: torch.Tensor, candidate_docs: torch.Tensor
    ) -> torch.Tensor:
        """Get the action (slate) of the agent"""
        return self.slate_gen_func(state, candidate_docs)
