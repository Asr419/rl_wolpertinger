import torch
import torch.distributions as dist

class DocumentSampler:
    def __init__(self, num_docs, num_topics, num_low_quality_topics, doc_length, mean_quality_low=-1.5, mean_quality_high=1.5, quality_std=1.0):
        self.num_docs_per_topic = num_docs // num_topics
        self.num_low_quality_topics = num_low_quality_topics
        self.num_high_quality_topics = num_topics - num_low_quality_topics
        self.num_docs_low_quality = self.num_docs_per_topic * self.num_low_quality_topics
        self.num_docs_high_quality = self.num_docs_per_topic * self.num_high_quality_topics
        self.doc_length = doc_length
        
        # Sample quality means for low and high quality topics
        self.mean_quality = torch.cat([
            torch.ones(self.num_docs_low_quality // self.num_docs_per_topic) * mean_quality_low,
            torch.ones(self.num_docs_high_quality // self.num_docs_per_topic) * mean_quality_high
        ])
        
        # Sample topic vectors for each document
        self.topics = torch.cat([
            torch.randint(low=0, high=self.num_low_quality_topics, size=(self.num_docs_low_quality // self.num_docs_per_topic,)),
            torch.randint(low=self.num_low_quality_topics, high=num_topics, size=(self.num_docs_high_quality // self.num_docs_per_topic,)),
        ])
        self.topic_vectors = torch.zeros(num_docs, num_topics)
        for i in range(num_docs):
            topic = self.topics[i]
            if topic < self.num_low_quality_topics:
                doc_index = i % self.num_docs_per_topic
                self.topic_vectors[i, topic] = 1
            else:
                doc_index = i % self.num_docs_per_topic
                self.topic_vectors[i, topic - self.num_low_quality_topics] = 1


    
    def sample_document(self):
        # Sample a topic
        topic = self.topic_dist.sample()
        
        # Sample a document from the selected topic
        doc_index = torch.randint(self.num_docs_per_topic, size=(1,)).item()
        doc = torch.zeros(self.num_topics)
        doc[(topic - self.num_low_quality_topics) * self.num_docs_per_topic + doc_index] = 1
        
        
        # Sample the quality of the document
        quality = self.quality_dists[topic].sample()
        
        return doc, quality
    
    def sample_documents(self, num_samples):
        docs = torch.zeros(num_samples, self.num_topics)
        qualities = torch.zeros(num_samples)
        for i in range(num_samples):
            doc, quality = self.sample_document()
            docs[i] = doc
            qualities[i] = quality
        return docs, qualities


class DocDataset(torch.utils.data.Dataset):
    def __init__(self, doc_sampler, doc_length):
        self.doc_sampler = doc_sampler
        self.doc_length = doc_length

    def __len__(self):
        return self.doc_sampler.num_docs

    def __getitem__(self, idx):
        # Get topic vector and quality for document at index idx
        topic_vec = self.doc_sampler.topic_vectors[idx]
        quality = self.doc_sampler.quality_values[idx]

        # Generate document based on topic vector and quality
        document = self.doc_sampler.generate_document(topic_vec, quality, self.doc_length)

        return document