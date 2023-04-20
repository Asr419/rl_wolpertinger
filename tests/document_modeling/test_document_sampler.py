from rl_recsys.document_modeling.document_sampler import DocumentSampler
from rl_recsys.document_modeling.document_sampler import DocDataset
import torch
from torch.utils.data import DataLoader

def test_document_sampler():
    num_docs = 1000
    num_topics = 20
    num_low_quality_topics = 14
    doc_length = 50
    mean_quality_low = -1.5
    mean_quality_high = 1.5
    quality_std = 1.0

    # Create document sampler
    doc_sampler = DocumentSampler(num_docs, num_topics, num_low_quality_topics, doc_length, mean_quality_low, mean_quality_high, quality_std)

    # Check that the topic vectors sum to 1 for each document
    topic_sums = torch.sum(doc_sampler.topic_vectors, dim=1)
    assert torch.allclose(topic_sums, torch.ones(num_docs))

    # Check that the topic vectors are one-hot encoded for each document
    assert torch.all(torch.eq(torch.sum(doc_sampler.topic_vectors, dim=1), torch.ones(num_docs)))

    # Create a DataLoader with batch size 32
    batch_size = 32
    doc_dataset = DocDataset(doc_sampler, doc_length)

    doc_loader = DataLoader(doc_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through a few batches of documents and check their shapes
    for batch in doc_loader:
        assert batch.shape == (batch_size, doc_length, num_topics)