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

    def search(self, embedding, k, verbose):
        scores, samples = self.dataset.get_nearest_examples('embedding', embedding, k=k)
        results_df = pd.DataFrame.from_dict(samples)
        results_df['score'] = scores
        results_df.sort_values('score', ascending=False, inplace=True)
        results_df.reset_index(inplace=True)
        results_df.drop(['timestamp','embedding', 'index'], axis=1, inplace=True)
        results_df = results_df.reindex(columns=['score', 'text', 'url'])
        if verbose:
            for _, row in results_df.iterrows():
                print(148 * '-')
                print(f'Score: {row.score}')
                print(f'URL: {row.url}')
                print(f'Text: {row.text}')
        return results_df
    

class PineconBackend(DataBackend):
    def dataset_upsert(self, dataset, batch_size, upsert_minibatch_size=1000, text_size=100):
        assert text_size < 20000, 'text_size should be less then 20000'
        dataset_size = len(dataset)
        for batch_start in range(0, dataset_size, batch_size):
            batch_end = min(batch_start + batch_size, dataset_size)
            data = dataset[batch_start: batch_end] 
            ids = [str(hash(url)) for url in data['url']]
            vecs = data['embedding']
            metadata = [{'url': row[0], 'text': row[1][:text_size]} for row in zip(data['url'], data['text'])]
            to_upsert = list(zip(ids, vecs, metadata))
            self.index.upsert(vectors=to_upsert, batch_size=upsert_minibatch_size)

    def __init__(self, dataset=None, api_key=None, environment='gcp-starter', index_name=None, metric='euclidian', batch_size=50000):
        self.api_key = api_key
        self.environment = environment
        self.batch_size=batch_size
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            dimension = len(dataset[0]['embedding'])
            shards = math.ceil(dataset.size_in_bytes / 1024 ** 3)
            pinecone.create_index(index_name, dimension=dimension, shards=shards, metric=metric)
        self.index = pinecone.Index(index_name)
        if dataset is not None:
            self.dataset_upsert(dataset, batch_size)
        
    def add_data(self, dataset):
        self.dataset_upsert(dataset, batch_size=self.batch_size)

    def search(self, embedding, k, verbose):
        embedding = embedding.tolist()
        answer = self.index.query(embedding, top_k=k, include_values=False ,include_metadata=True)
        results_dict = {
            'score':[row['score'] for row in answer['matches']],
            'text': [row['metadata']['text'] for row in answer['matches']],
            'url': [row['metadata']['url'] for row in answer['matches']],
        }
        results_df = pd.DataFrame.from_dict(results_dict)
        results_df.sort_values('score', ascending=False, inplace=True)
        if verbose:
            for _, row in results_df.iterrows():
                print(148 * '-')
                print(f'Score: {row.score}')
                print(f'URL: {row.url}')
                print(f'Text: {row.text}')
        return results_df