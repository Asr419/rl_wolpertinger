import numpy as np


class Node:
    def __init__(self, belief_state):
        self.belief_state = belief_state
        self.visits = 0
        self.value = 0
        self.children = {}

    def select(self, c):
        max_ucb = float("-inf")
        selected_node = None
        for a, child in self.children.items():
            ucb = child.value / child.visits + c * np.sqrt(
                np.log(self.visits) / child.visits
            )
            if ucb > max_ucb:
                max_ucb = ucb
                selected_node = child
        return selected_node

    def add_child(self, action, node):
        self.children[action] = node

    def expand(self, action, P, O):
        next_belief_state = P[action] @ self.belief_state
        next_belief_state /= np.sum(next_belief_state)
        next_node = Node(next_belief_state)
        return next_node

    def backup(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits


class PomcpUserModel:
    def __init__(self, A, S, O, P, c=1):
        self.A = A
        self.S = S
        self.O = O
        self.P = P
        self.c = c

        # Initialize the root node with a uniform prior over the states
        self.root = Node(np.ones(len(S)) / len(S))

    def plan(self, n_iterations=1000):
        for i in range(n_iterations):
            # Selection: Select a node to expand using the UCT (Upper Confidence Bound for Trees) algorithm
            node = self.root.select(self.c)

            # Expansion: Add a new child node to the selected node for each possible action
            for a in self.A:
                new_node = node.expand(a, self.P, self.O)
                node.add_child(a, new_node)

            # Simulation: Simulate a trajectory from the newly added child nodes
            reward = self.rollout(new_node)

            # Backpropagation: Update the value estimates of all nodes visited during the selection and expansion stages
            node.backup(reward)

    def rollout(self, node):
        # Use a rollout policy to simulate a trajectory from the current node
        s = self.sample_belief_state(node.belief_state)
        reward = 0
        discount = 1

        while not s.is_terminal():
            a = self.rollout_policy(s)
            o = s.observe(a).sample()
            reward += discount * s.reward(a, o)
            discount *= s.discount_factor
            s = s.update(a, o)

        return reward

    def sample_belief_state(self, belief_state):
        # Sample a state from the belief state
        state_idx = np.random.choice(len(self.S), p=belief_state)
        return self.S[state_idx]

    def rollout_policy(self, state):
        # Use a random policy for the rollout phase
        return np.random.choice(self.A)

    def update_belief_state(self, a, o):
        # Update the belief state using Bayes' rule
        new_belief_state = self.O[a][:, o] @ self.P[a] @ self.root.belief_state
        new_belief_state /= np.sum(new_belief_state)
        self.root.belief_state = new_belief_state
        self.root.visits = 0
        self.root.value = 0


belief = pomcp_user_model.root.belief_state
