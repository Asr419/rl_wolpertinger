from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATS_PREPROCESSING = ["year", "popularity", "key", "tempo", "duration_ms"]
FINAL_FEATS = [
    "song_id",
    "year",
    "popularity",
    "valence",
    "danceability",
    "loudness",
    "speechiness",
    "acousticness",
    "liveness",
    "key",
    "mode",
    "tempo",
    "instrumentalness",
    "energy",
    "duration_ms",
]

DATASET_NAME = "Spotify"
DATA_PATH = Path(Path.home() / "rsys_data")

if __name__ == "__main__":
    DATA_PATH = Path(Path.home() / "rsys_data")
    dataset_path = DATA_PATH / DATASET_NAME
    doc_feat = pd.read_feather(dataset_path)

    scaler = RobustScaler()

    print("Normalizing features using: {}".format(scaler.__class__.__name__))

    for feat in FEATS_PREPROCESSING:
        doc_feat[feat] = scaler.fit_transform(doc_feat[[feat]])

    # adding song_id as integer
    doc_feat["song_id"] = np.arange(len(doc_feat))

    # filtering on relevant features
    doc_feat = doc_feat[FINAL_FEATS]

    # saving preproessed features
    save_path = DATA_PATH / "prep_spotify.feather"
    doc_feat.to_feather(save_path)
    print("Preprocessed features saved to: {}".format(save_path))
