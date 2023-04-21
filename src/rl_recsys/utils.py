from pathlib import Path

import pandas as pd

# TODO: to be moved in the env file
DATA_PATH = Path(Path.home() / "rsys_data")
SAVE_PATH = DATA_PATH / "prep_spotify.feather"
SAVE_PATH_TOPIC = DATA_PATH / "prep_topic.feather"


def load_spotify_data() -> pd.DataFrame:
    """Load spotify data."""
    return pd.read_feather(SAVE_PATH)

def load_topic_data()-> pd.DataFrame:
    """Load topic data."""
    return pd.read_feather(SAVE_PATH_TOPIC)


if __name__ == "__main__":
    print(load_spotify_data())
    print(load_topic_data())
