from ast import Tuple
from pathlib import Path

import pandas as pd
import torch
import torch.distributions as dist


class DocumentSampler:
    def __init__(
        self,
        num_topics: int,
        num_low_quality_topics: int,
        num_docs_per_topic: int,
        doc_length: int,
        quality_variance: float,
    ):
        self.num_topics = num_topics
        self.num_low_quality_topics = num_low_quality_topics
        self.num_high_quality_topics = num_topics - num_low_quality_topics
        self.num_docs_per_topic = num_docs_per_topic
        self.doc_length = doc_length
        self.quality_variance = quality_variance

        # Initialize the topic distribution over documents
        low_quality_topic_probs = (
            torch.ones(self.num_low_quality_topics) / self.num_low_quality_topics
        )
        high_quality_topic_probs = (
            torch.ones(self.num_high_quality_topics) / self.num_high_quality_topics
        )

        self.topic_dist = dist.Categorical(
            torch.cat([low_quality_topic_probs, high_quality_topic_probs])
        )

        # Initialize the quality distribution for each document
        low_quality_topic_means = torch.linspace(-3, 0, num_low_quality_topics)
        high_quality_topic_means = torch.linspace(0, 3, self.num_high_quality_topics)
        self.topic_means = torch.cat(
            [low_quality_topic_means, high_quality_topic_means]
        )
        self.quality_dists: list[dist.Normal] = []
        for t in range(num_topics):
            mean_quality = self.topic_means[t]
            self.quality_dists.append(dist.Normal(mean_quality, quality_variance))

    def sample_document(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Sample a topic
        topic = self.topic_dist.sample()

        # Sample a document from the selected topic
        doc_index = torch.randint(self.num_docs_per_topic, size=(1,)).item()
        doc = torch.zeros(self.num_topics)

        doc[topic] = 1

        # Sample the quality of the document
        quality = self.quality_dists[topic].sample()

        return doc, quality, self.doc_length

    def sample_documents(
        self, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        docs = torch.zeros(num_samples, self.num_topics)
        qualities = torch.zeros(num_samples)
        doc_length = torch.zeros(num_samples)
        for i in range(num_samples):
            doc, quality, doc_len = self.sample_document()
            docs[i] = doc
            qualities[i] = quality
            doc_length[i] = doc_len

        return docs, qualities, doc_length
