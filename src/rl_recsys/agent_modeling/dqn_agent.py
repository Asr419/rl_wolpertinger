import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_recsys.agent_modeling.agent import AbstractBeliefAgent, AbstractSlateAgent

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNnet(nn.Module):
    def __init__(self, input_size, output_size=1):
        # todo: change intra dimensions
        super(DQNnet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent(AbstractSlateAgent, nn.Module):
    def __init__(
        self, slate_gen_func, input_size: int, output_size: int, tau: float
    ) -> None:
        # init super classes
        AbstractSlateAgent.__init__(self, slate_gen_func)
        nn.Module.__init__(self)
        self.tau = tau

        # init DQN nets
        self.policy_net = DQNnet(input_size=input_size, output_size=output_size)
        self.target_net = DQNnet(input_size=input_size, output_size=output_size)
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

    def compute_candidates_q_values(
        self, state: torch.Tensor, candidate_docs_repr: torch.Tensor
    ) -> torch.Tensor:
        ...
        # compute q-values
        return self.policy_net(state)

    def get_action(
        self, state: torch.Tensor, candidate_docs: torch.Tensor
    ) -> torch.Tensor:
        # compute q-values
        ...
        slate = self.slate_gen_func()
        pass

    def get_target_q_values(self, next_state, reward):
        pass
