import math
from abc import ABC, abstractmethod
import pandas as pd
from datasets import concatenate_datasets
import pinecone

class DataBackend(ABC):
    @abstractmethod
    def add_data(self):
        pass

    @abstractmethod
    def search(self):
        pass


class LocalBackend(DataBackend):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.add_faiss_index('embedding')
        
    def add_data(self, dataset):
        self.dataset = concatenate_datasets([self.dataset, dataset])
        self.dataset.add_faiss_index('embedding')

    def search(self, embedding, verbose=True):
        scores, samples = self.dataset.get_nearest_examples('embedding', embedding, k=5)
        results_df = pd.DataFrame.from_dict(samples)
        results_df['scores'] = scores
        results_df.sort_values('scores', ascending=False, inplace=True)
        if verbose:
            for _, row in results_df.iterrows():
                print(148 * '-')
                print(f'Scores: {row.scores}')
                print(f'URL: {row.url}')
                print(f'Text: {row.text}')
        return results_df
    

class PineconBackend(DataBackend):
    def dataset_upsert(self, dataset, batch_size):
        dataset_size = len(dataset)
        for batch_start in range(0, dataset_size, batch_size):
            batch_end = min(batch_start + batch_size, dataset_size)
            data = dataset[batch_start: batch_end] 
            ids = [str(hash(url)) for url in data['url']]
            vecs = data['embedding']
            metadata = [{'url': row[0], 'text': row[1][:20000]} for row in zip(data['url'], data['text'])]
            to_upsert = list(zip(ids, vecs, metadata))
            self.index.upsert(vectors=to_upsert, batch_size=1000)

    def __init__(self, dataset=None, api_key=None, environment='gcp-starter', index_name=None, batch_size=50000):
        self.api_key = api_key
        self.environment = environment
        dimension = len(dataset[0]['embedding'])
        shards = math.ceil(dataset.size_in_bytes / 1024 ** 3)
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=dimension, shards=shards)
        self.index = pinecone.Index(index_name)
        if dataset is not None:
            self.dataset_upsert(dataset, batch_size)
        
    def add_data(self, dataset, batch_size=50000):
        self.dataset_upsert(dataset, batch_size)

    def search(self, embedding, verbose=True):
        rez = self.index.query(embedding, top_k=5, include_values=False ,include_metadata=True)