from pathlib import Path

import pandas as pd

from rl_recsys.document_modeling.document_sampler import DocumentSampler

NUM_TOPICS = 20
NUM_SAMPLES = 100
NUM_LOW_QUALITY_TOPICS = 14
NUM_DOCS_PER_TOPIC = NUM_SAMPLES // NUM_TOPICS
DOC_LENGTH = 4


if __name__ == "__main__":
    assert (
        NUM_TOPICS > NUM_LOW_QUALITY_TOPICS
    ), "Number of topics must be greater than number of low quality topics"

    # creating a simulated item catalogue and save it
    sampler = DocumentSampler(
        num_topics=NUM_TOPICS,
        num_low_quality_topics=NUM_LOW_QUALITY_TOPICS,
        num_docs_per_topic=NUM_DOCS_PER_TOPIC,
        doc_length=DOC_LENGTH,
        quality_variance=1.0,
    )
    docs, qualities, doc_length = sampler.sample_documents(num_samples=NUM_SAMPLES)

    df = pd.DataFrame(docs.numpy())

    # add colunmns to the dataframe
    df.insert(0, "doc_id", range(0, len(df)))
    df.columns = ["doc_id"] + [f"topic_{i}" for i in range(1, NUM_TOPICS + 1)]
    df["doc_length"] = doc_length.numpy()
    df["doc_quality"] = qualities.numpy()

    DATASET_NAME = "Topic_Dataset"
    DATA_PATH = Path(Path.home() / "rsys_data")
    save_path = DATA_PATH / "prep_topic.feather"
    df.to_feather(save_path)
    print("Preprocessed features saved to: {}".format(save_path))
