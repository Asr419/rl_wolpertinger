import argparse
import configparser
import os
import pickle
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from rl_recsys.agent_modeling.dqn_agent import DQNAgent, ReplayMemoryDataset, Transition
from rl_recsys.agent_modeling.slate_generator import (  # DiverseSlateGenerator,; GreedySlateGenerator,; OptimalSlateGenerator,
    RandomSlateGenerator,
    TopKSlateGenerator,
)
from rl_recsys.agent_modeling.wp_agent import WolpertingerActor, ActorAgent
from rl_recsys.agent_modeling.wp_slate_agent import (
    WolpertingerActorSlate,
    ActorAgentSlate,
)
from rl_recsys.document_modeling.document_sampler import DocumentSampler
from rl_recsys.simulation_environment.environment import SlateGym
from rl_recsys.user_modeling.choice_model import (
    CosineSimilarityChoiceModel,
    DotProductChoiceModel,
)
from rl_recsys.user_modeling.features_gen import (
    NormalUserFeaturesGenerator,
    UniformFeaturesGenerator,
)
from rl_recsys.user_modeling.response_model import (
    CosineResponseModel,
    DotProductResponseModel,
    WeightedDotProductResponseModel,
)
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import ObservableUserState
from rl_recsys.utils import save_run, save_run_wa

class_name_to_class = {
    "ObservedUserState": ObservableUserState,
    "DotProductChoiceModel": DotProductChoiceModel,
    "DotProductResponseModel": DotProductResponseModel,
    "TopKSlateGenerator": TopKSlateGenerator,
    "RandomSlateGenerator": RandomSlateGenerator,
    # "DiverseSlateGenerator": DiverseSlateGenerator,
    # "GreedySlateGenerator": GreedySlateGenerator,
    # "OptimalSlateGenerator": OptimalSlateGenerator,
    "WeightedDotProductResponseModel": WeightedDotProductResponseModel,
}
load_dotenv()
