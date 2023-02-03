import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats
import random
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from pathlib import Path
from sklearn import preprocessing
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class LTSDocument(document.AbstractDocument):
    def __init__(
        self,
        doc_id,
        year,
        name,
        artists,
        popularity,
        valence,
        song_id,
        danceability,
        loudness,
        speechiness,
        acousticness,
        liveness,
        label,
    ):
        self.year = year
        self.name = name
        self.artists = artists
        self.popularity = popularity
        self.valence = valence
        self.song_id = song_id
        self.danceability = danceability
        self.loudness = loudness
        self.speechiness = speechiness
        self.acousticness = acousticness
        self.liveness = liveness
        self.label = label

        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return self.label

        # return np.array([self.genre])

    @staticmethod
    def observation_space():
        return spaces.Discrete(10)
        # return spaces.Box(shape=(1,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        # return f"{self._doc_id}"
        return "Music {} with genre {}.".format(
            self._doc_id,
            self.year,
            self.name,
            self.artists,
            self.popularity,
            self.valence,
            self.song_id,
            self.danceability,
            self.loudness,
            self.speechiness,
            self.acousticness,
            self.liveness,
            self.label,
        )


class LTSDocumentSampler(document.AbstractDocumentSampler):
    DATASET_NAME = "Spotify"
    _DATA_PATH = Path(Path.home() / "rsys_data")
    _DATASET_PATH = _DATA_PATH / DATASET_NAME
    songs = pd.read_feather(_DATASET_PATH)

    def __init__(self, doc_ctor=LTSDocument, **kwargs):
        super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._music_count = 0

    def sample_document(self, songs=songs):
        s = randrange(len(songs.index))
        doc_features = {}
        doc_features["doc_id"] = self._music_count
        doc_features["year"] = songs.loc[[s]].year
        doc_features["name"] = songs.loc[[s]].name
        doc_features["artists"] = songs.loc[[s]].artists
        doc_features["popularity"] = songs.loc[[s]].popularity
        doc_features["valence"] = songs.loc[[s]].valence
        doc_features["song_id"] = songs.loc[[s]].song_id
        doc_features["danceability"] = songs.loc[[s]].danceability
        doc_features["acousticness"] = songs.loc[[s]].acousticness
        doc_features["liveness"] = songs.loc[[s]].liveness
        doc_features["label"] = songs.loc[[s]].label
        doc_features["loudness"] = songs.loc[[s]].loudness
        doc_features["speechiness"] = songs.loc[[s]].speechiness
        self._music_count += 1
        return self._doc_ctor(**doc_features)


if __name__ == "__main__":
    spotify_data = pd.read_csv("src/recsim/data/data.csv")
    genre_data = pd.read_csv("src/recsim/data/data_by_genres.csv")
    data_by_year = pd.read_csv("src/recsim/data/data_by_year.csv")

    loudness = spotify_data[["loudness"]].values
    min_max_scaler = preprocessing.MinMaxScaler()
    loudness_scaled = min_max_scaler.fit_transform(loudness)
    spotify_data["loudness"] = pd.DataFrame(loudness_scaled)
    songs_features = spotify_data[
        ["danceability", "loudness", "speechiness", "acousticness", "liveness"]
    ]

    # Sum_of_squared_distances = []
    # K = range(1, 15)
    # for k in K:
    #     km = KMeans(n_clusters=k)
    #     with threadpool_limits(user_api="openmp", limits=2):
    #         km = km.fit(songs_features)
    #     Sum_of_squared_distances.append(km.inertia_)
    kmeans = KMeans(n_clusters=4)
    with threadpool_limits(user_api="openmp", limits=2):
        kmeans.fit(songs_features)
    y_kmeans = kmeans.predict(songs_features)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(songs_features)

    pc = pd.DataFrame(principal_components)
    pc["label"] = y_kmeans
    pc.columns = ["x", "y", "label"]

    # plot data with seaborn
    cluster = sns.lmplot(
        data=pc, x="x", y="y", hue="label", fit_reg=False, legend=True, legend_out=True
    )

    tsne = TSNE(n_components=2, perplexity=50)

    tsne_components = tsne.fit_transform(songs_features)

    ts = pd.DataFrame(tsne_components)
    ts["label"] = y_kmeans
    ts.columns = ["x", "y", "label"]

    # plot data with seaborn
    cluster = sns.lmplot(
        data=ts, x="x", y="y", hue="label", fit_reg=False, legend=True, legend_out=True
    )

    spotify_data["label"] = y_kmeans

    # shuffle dataset

    # songs = spotify_data.sample(frac=1)
    spotify_data["label"].value_counts()

    songs = spotify_data[
        [
            "year",
            "name",
            "artists",
            "popularity",
            "valence",
            "id",
            "danceability",
            "loudness",
            "speechiness",
            "acousticness",
            "liveness",
            "label",
        ]
    ]

    DATASET_NAME = "Spotify"
    songs.rename(columns={"id": "song_id"}, inplace=True)
    _DATA_PATH = Path(Path.home() / "rsys_data")
    _DATASET_PATH = _DATA_PATH / DATASET_NAME
    songs.to_feather(_DATASET_PATH)
