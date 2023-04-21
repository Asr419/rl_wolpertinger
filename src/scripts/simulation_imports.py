import argparse
import configparser
import os
import pickle
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random
import numpy as np

import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from dotenv import load_dotenv
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
from rl_recsys.agent_modeling.wp_agent import WolpertingerActor
from rl_recsys.belief_modeling.belief_model import NNBeliefModel
from rl_recsys.belief_modeling.history_model import (
    AvgHistoryModel,
    GRUModel,
    LastObservationModel,
)
from rl_recsys.document_modeling.documents_catalogue import DocCatalogue, TopicDocCatalogue
from rl_recsys.retrieval import ContentSimilarityRec
from rl_recsys.simulation_environment.environment import MusicGym
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
)
from rl_recsys.user_modeling.user_model import UserSampler
from rl_recsys.user_modeling.user_state import AlphaIntentUserState
from rl_recsys.utils import load_spotify_data
from rl_recsys.utils import load_topic_data

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
    "LastObservationModel": LastObservationModel,
    "NNBeliefModel": NNBeliefModel,
}
load_dotenv()
