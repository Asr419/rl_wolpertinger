from pathlib import Path

import numpy as np
import pandas as pd

from rl_recsys.document_modeling.documents_catalogue import DocCatalogue
from rl_recsys.utils import load_spotify_data


def test_doc_catalogue():
    # load spotify data
    data_df = load_spotify_data()

    doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")

    doc_id = 0
    doc_feature = doc_catalogue.get_doc_features(doc_id=doc_id)
    assert doc_feature.shape == (14,)

    doc_features = doc_catalogue.get_all_item_features()
    assert doc_features.shape == (len(doc_features), 14)

    doc_ids = np.array([0, 1, 2])
    doc_features = doc_catalogue.get_docs_features(doc_ids=doc_ids)
    assert doc_features.shape == (len(doc_ids), 14)
