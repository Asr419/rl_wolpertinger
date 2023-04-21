import torch
import torch.distributions as dist
from pathlib import Path
import pandas as pd

class DocumentSampler:
    def __init__(self, num_topics, num_low_quality_topics, num_docs_per_topic, doc_length, quality_variance):
        self.num_topics = num_topics
        self.num_low_quality_topics = num_low_quality_topics
        self.num_high_quality_topics = num_topics - num_low_quality_topics
        self.num_docs_per_topic = num_docs_per_topic
        self.doc_length = doc_length
        self.quality_variance = quality_variance
        
        # Initialize the topic distribution over documents
        low_quality_topic_probs = torch.ones(num_low_quality_topics) / num_low_quality_topics
        high_quality_topic_probs = torch.ones(self.num_high_quality_topics) / self.num_high_quality_topics
        self.topic_dist = dist.Categorical(torch.cat([low_quality_topic_probs, high_quality_topic_probs]))
        
        # Initialize the mean quality of each topic
        low_quality_topic_means = torch.linspace(-3, 0, num_low_quality_topics)
        high_quality_topic_means = torch.linspace(0, 3, self.num_high_quality_topics)
        self.topic_means = torch.cat([low_quality_topic_means, high_quality_topic_means])
        
        # Initialize the quality distribution for each document
        self.quality_dists = []
        for t in range(num_topics):
            mean_quality = self.topic_means[t]
            self.quality_dists.append(dist.Normal(mean_quality, quality_variance))
    
    def sample_document(self):
        # Sample a topic
        topic = self.topic_dist.sample()
        
        # Sample a document from the selected topic
        doc_index = torch.randint(self.num_docs_per_topic, size=(1,)).item()
        doc = torch.zeros(self.num_topics)
        
        doc[topic]=1
        
        # Sample the quality of the document
        quality = self.quality_dists[topic].sample()
        
        return doc, quality, self.doc_length
    
    def sample_documents(self, num_samples):
        docs = torch.zeros(num_samples, self.num_topics)
        qualities = torch.zeros(num_samples)
        doc_length=torch.zeros(num_samples)
        for i in range(num_samples):
            doc, quality,doc_len = self.sample_document()
            docs[i] = doc
            qualities[i] = quality
            doc_length[i]=doc_len

        return docs, qualities,doc_length


if __name__ == "__main__":
    num_topics=20
    num_samples=50000
    sampler = DocumentSampler(num_topics=num_topics, num_low_quality_topics=14, num_docs_per_topic=300, doc_length=4, quality_variance=1.0)
    docs, qualities, doc_length = sampler.sample_documents(num_samples=num_samples)
    # print(docs.shape)
    # print(qualities.shape)
    # print(doc_length.shape)
    df = pd.DataFrame(docs.numpy())
    df.insert(0, 'doc_id', range(0, len(df)))
    df.columns =['doc_id']+ [f'topic_{i}' for i in range(1, num_topics+1)]
    df['doc_length'] = doc_length.numpy()
    df['doc_quality'] = qualities.numpy()
   
    DATASET_NAME = "Topic_Dataset"
    DATA_PATH = Path(Path.home() / "rsys_data")
    save_path = DATA_PATH / "prep_topic.feather"
    df.to_feather(save_path)
    print("Preprocessed features saved to: {}".format(save_path))

