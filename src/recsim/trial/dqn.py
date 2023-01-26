import random as rand
from math import sqrt
import numpy as np
import pandas as pd

from gym import Env
from gym.spaces import Discrete, Box
from numpy import int64

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from gym.spaces import Box, Discrete
from tqdm import tqdm


from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


class CustomEnv1(Env):
    def __init__(self):
        self.i = 0
        self.action_space = Discrete(100)
        self.observation_space = Box(low=0, high=99, shape=(1, 10), dtype=int64)
        self.state = self.observation_space.sample().ravel()

    def step(self, action):
        self.i += 1
        for j in range(self.i, 10):
            self.state[j] = -1
        self.next_state = self.observation_space.sample().ravel()
        # self.next_state[0] = self.state[1]
        # self.next_state[1]= action

        if action in self.state:
            self.state[self.i] = action
            reward = -10
        else:
            self.state[self.i] = action
            reward = CustomEnv1.getReward1(ru, D, self.state, action)

        if self.i == 9:
            print(self.state)
            done = True
        else:
            done = False
        info = {}

        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        self.i = 0
        self.state = self.observation_space.sample().ravel()
        return self.state

    def getReward1(ru, D, state, action):
        nitems = len(ru)
        dist = D.flatten()
        try:
            reward = ru[action][0]
            for i in range(0, len(state)):
                if state[i] == -1:
                    reward += 0
                    break
                else:
                    reward += (1 / ((len(state) - i) + 1)) * dist[
                        (state[i]) * nitems + action
                    ]
        except IndexError:
            reward = 1.5
        return reward


if __name__ == "__main__":
    nusers = 100
    nitems = 100
    k = 10

    pu = np.random.rand(k, 1)

    Q = np.random.rand(nitems, k)
    nQ = np.dot(Q, Q.T)

    ru = np.dot(Q, pu)
    D = np.diag(nQ) + np.diag(nQ.T) - 2 * nQ

    env = CustomEnv1()

    episodes = 10  # 20 shower episodes
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))

    states = np.shape(env.observation_space)
    actions = env.action_space.n

    def build_model(states, actions):
        model = Sequential()
        model.add(Dense(24, activation="relu", input_shape=states))
        model.add(Dense(24, activation="relu"))
        model.add(Flatten())
        model.add(Dense(actions, activation="linear"))
        return model

    model = build_model(states, actions)

    def build_agent(model, actions):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(
            model=model,
            memory=memory,
            policy=policy,
            nb_actions=actions,
            nb_steps_warmup=10,
            target_model_update=1e-2,
            gamma=0.1,
        )
        return dqn

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    history = dqn.fit(env, nb_steps=50001, visualize=False, verbose=1)

    results = dqn.test(env, nb_episodes=30, visualize=False)
    print(np.mean(results.history["episode_reward"]))
