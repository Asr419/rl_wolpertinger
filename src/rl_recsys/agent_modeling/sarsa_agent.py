import random
import time
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from rl_recsys.agent_modeling.agent import AbstractSlateAgent


class ReplayMemoryDataset(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple(
            "Transition",
            (
                "state",
                "selected_doc_feat",
                "reward",
                "next_state",
                "next_selected_doc_feat",
            ),
        )

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def __getitem__(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)


def replay_memory_collate_fn(batch):
    Transition = namedtuple(
        "Transition",
        (
            "state",
            "selected_doc_feat",
            "reward",
            "next_state",
            "next_selected_doc_feat",
        ),
    )

    transitions_batch = Transition(*zip(*batch))

    state_batch = torch.stack(transitions_batch.state)
    selected_doc_feat_batch = torch.stack(transitions_batch.selected_doc_feat)
    next_state_batch = torch.stack(transitions_batch.next_state)
    # candidates_batch = torch.stack(transitions_batch.candidates_docs)
    reward_batch = torch.stack(transitions_batch.reward)
    next_selected_doc_feat_batch = torch.stack(transitions_batch.next_selected_doc_feat)

    return (
        state_batch,
        selected_doc_feat_batch,
        reward_batch,
        next_state_batch,
        next_selected_doc_feat_batch,
    )


class SARSAnet(nn.Module):
    def __init__(self, input_size, hidden_dims: list[int], output_size=1):
        # todo: change intra dimensions
        super(SARSAnet, self).__init__()

        self.layers = nn.ModuleList()

        # create layers based on hidden_dims
        prev_dim = input_size
        for dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        # add last layer
        self.layers.append(nn.Linear(prev_dim, output_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x)
        return x


class SARSAAgent(AbstractSlateAgent, nn.Module):
    def __init__(
        self,
        slate_gen,
        input_size: int,
        output_size: int,
        hidden_dims: list[int] = [28, 14, 7],
        tau: float = 0.001,
    ) -> None:
        # init super classes
        AbstractSlateAgent.__init__(self, slate_gen)
        nn.Module.__init__(self)
        self.tau = tau

        # init DQN nets
        self.policy_net = SARSAnet(
            input_size=input_size, output_size=output_size, hidden_dims=hidden_dims
        )

        # note that the target network is not updated during training
        self.target_net = SARSAnet(
            input_size=input_size, output_size=output_size, hidden_dims=hidden_dims
        )
        self.target_net.requires_grad_(False)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target_network(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def compute_q_values(
        self,
        state: torch.Tensor,
        candidate_docs_repr: torch.Tensor,
        use_policy_net: bool = True,
    ) -> torch.Tensor:
        # concatenate state and candidate docs
        input1 = torch.cat([state, candidate_docs_repr], dim=1)
        # [num_candidate_docs, 1]
        if use_policy_net:
            q_val = self.policy_net(input1)
        else:
            q_val = self.target_net(input1)
        return q_val

    def compute_target_q_values(self, next_state, reward):
        pass
