import os
import numpy as np
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple
from pathlib import Path
import random
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
import p4


class LTSUserState(user.AbstractUserState):
    def __init__(
        self,
        age,
        gender,
        valence,
        danceability,
        loudness,
        speechiness,
        acousticness,
        liveness,
        net_genre_exposure,
        sensitivity,
        time_budget,
        label,
        mode,
        key,
        duration_ms,
        tempo,
        energy,
        instrumentalness,
        observation_noise_stddev=0.1,
    ):
        ## Transition model parameters
        ## State variables
        ##############################
        self.age = age
        self.gender = gender
        self.sensitivity = sensitivity
        self.valence = valence
        self.danceability = danceability

        ## Engagement parameters
        self.loudness = loudness
        self.speechiness = speechiness
        self.acousticness = acousticness
        self.liveness = liveness

        self.label = label
        self.mode = mode
        self.key = key
        self.duration_ms = duration_ms
        self.tempo = tempo
        self.energy = energy
        self.instrumentalness = instrumentalness
        self.net_genre_exposure = net_genre_exposure
        self.satisfaction = 1 / (1 + np.exp(-sensitivity * net_genre_exposure))
        self.time_budget = time_budget

        # Noise
        self._observation_noise = observation_noise_stddev

    def create_observation(self):
        """User's state is not observable."""
        clip_low, clip_high = (
            -1.0 / (1.0 * self._observation_noise),
            1.0 / (1.0 * self._observation_noise),
        )
        noise = stats.truncnorm(
            clip_low, clip_high, loc=0.0, scale=self._observation_noise
        ).rvs()
        noisy_sat = self.satisfaction + noise
        return np.array(
            [
                noisy_sat,
            ]
        )

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(1,), dtype=np.float32, low=-2.0, high=2.0)

    # scoring function for use in the choice model -- the user is more likely to
    # click on more chocolatey content.
    def score_document(self, doc_obs):
        return 1 - doc_obs


class LTSStaticUserSampler(user.AbstractUserSampler):
    _state_parameters = None

    def __init__(
        self,
        user_ctor=LTSUserState,
        sensitivity=0.01,
        time_budget=30,
        #  age=np.random.random_integers(20,60),
        #  gender=np.random.random_integers(1),
        #  valence=np.random.uniform(0.0,1.0),
        #  danceability=np.random.normal(0.5373955347986852,0.17613721955546152),
        #  loudness=np.random.gumbel(0.8004193058243345,0.0690033070151354),
        #  speechiness=np.random.laplace(0.045,0.06335336735949558),
        #  acousticness=np.random.uniform(0.0,0.996),
        #  liveness=np.random.laplace(0.136,0.11092283130094402),
        #  mood=np.random.random_integers(3),
        **kwargs
    ):
        self._state_parameters = {
            "sensitivity": sensitivity,
            "time_budget": time_budget,
        }  #'age' : age,
        #                           'gender_group' : gender_group,
        #                           'valence': valence,
        #                           'danceability': danceability,
        #                           'loudness': loudness,
        #                           'speechiness': speechiness,
        #                           'acousticness': acousticness,
        #                           'liveness': liveness,
        #                           'mood': mood

        super(LTSStaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        self._state_parameters["age"] = np.random.random_integers(20, 60)
        self._state_parameters["gender"] = np.random.random_integers(2)
        self._state_parameters["valence"] = np.random.uniform(0.0, 1.0)
        self._state_parameters["danceability"] = np.random.normal(
            0.5373955347986852, 0.17613721955546152
        )
        self._state_parameters["loudness"] = np.random.gumbel(
            0.8004193058243345, 0.0690033070151354
        )
        self._state_parameters["speechiness"] = np.random.laplace(
            0.045, 0.06335336735949558
        )
        self._state_parameters["acousticness"] = np.random.uniform(0.0, 0.996)
        self._state_parameters["liveness"] = np.random.laplace(
            0.136, 0.11092283130094402
        )
        self._state_parameters["label"] = np.random.random_integers(4)
        self._state_parameters["mode"] = np.random.random_integers(2)
        self._state_parameters["key"] = np.random.random_integers(12)
        self._state_parameters["duration_ms"] = np.random.laplace(
            207467.0, 69419.7225773939
        )
        self._state_parameters["tempo"] = np.random.gumbel(
            102.26991914996023, 27.79478925686645
        )
        self._state_parameters["energy"] = np.random.uniform(0.0, 1.0)
        self._state_parameters["instrumentalness"] = np.random.exponential(0.167)

        starting_nke = self._rng.random_sample() - 0.5
        self._state_parameters["net_genre_exposure"] = starting_nke
        # starting_nke = ((self._rng.random_sample() - .5) *
        #                 (1 / (1.0 - self._state_parameters['memory_discount'])))
        # self._state_parameters['net_genre_exposure'] = starting_nke
        return self._user_ctor(**self._state_parameters)


if __name__ == "__main__":
    DATASET_NAME = "Spotify"
    _DATA_PATH = Path(Path.home() / "rsys_data")
    _DATASET_PATH = _DATA_PATH / DATASET_NAME
    songs = pd.read_feather(_DATASET_PATH)
    sampler = LTSStaticUserSampler()
    User = pd.DataFrame(
        columns=[
            "User_ID",
            "age",
            "gender",
            "valence",
            "danceability",
            "loudness",
            "speechiness",
            "acousticness",
            "liveness",
            "mood",
            "mode",
            "key",
            "duration_ms",
            "tempo",
            "energy",
            "instrumentalness",
        ]
    )
    # starting_nke = []

    for i in range(1000):
        sampled_user = sampler.sample_user()
        k = sampled_user
        User = User.append(
            {
                "User_ID": i,
                "age": k.age,
                "gender": k.gender,
                "valence": k.valence,
                "danceability": k.danceability,
                "loudness": k.loudness,
                "speechiness": k.speechiness,
                "acousticness": k.acousticness,
                "liveness": k.liveness,
                "mood": k.label,
                "mode": k.mode,
                "key": k.key,
                "duration_ms": k.duration_ms,
                "tempo": k.tempo,
                "energy": k.energy,
                "instrumentalness": k.instrumentalness,
            },
            ignore_index=True,
        )
    print(User)
