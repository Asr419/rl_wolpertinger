import argparse
import configparser
import os
from datetime import datetime
import shutil
import pickle
from collections import defaultdict
import pytorch_lightning as pl

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from rl_recsys.agent_modeling.agent import BeliefAgent
from rl_recsys.agent_modeling.dqn_agent import (
    DQNAgent,
    GruTransition,
    ReplayMemoryDataset,
    Transition,
)
from rl_recsys.agent_modeling.slate_generator import (
    DiverseSlateGenerator,
    GreedySlateGenerator,
    OptimalSlateGenerator,
    TopKSlateGenerator,
)
from rl_recsys.belief_modeling.history_model import AvgHistoryModel, GRUModel
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.simulation_environment.environment import MusicGym
from rl_recsys.user_modeling.choice_model import (
    CosineSimilarityChoiceModel,
    DotProductChoiceModel,
)
from rl_recsys.user_modeling.features_gen import NormalUserFeaturesGenerator
from rl_recsys.user_modeling.response_model import (
    CosineResponseModel,
    DotProductResponseModel,
)
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.utils import load_spotify_data

class_name_to_class = {
    "AlphaIntentUserState": AlphaIntentUserState,
    "DotProductChoiceModel": DotProductChoiceModel,
    "CosineResponseModel": CosineResponseModel,
    "CosineSimilarityChoiceModel": CosineSimilarityChoiceModel,
    "DotProductResponseModel": DotProductResponseModel,
    "AvgHistoryModel": AvgHistoryModel,
    "GRUModel": GRUModel,
    "TopKSlateGenerator": TopKSlateGenerator,
    "DiverseSlateGenerator": DiverseSlateGenerator,
    "GreedySlateGenerator": GreedySlateGenerator,
    "OptimalSlateGenerator": OptimalSlateGenerator,
}
