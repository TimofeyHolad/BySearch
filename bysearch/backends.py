from typing import Optional
import math
from abc import ABC, abstractmethod
import pandas as pd
from pandas import DataFrame
from datasets import concatenate_datasets, Dataset
import pinecone
import chromadb


class DataBackend(ABC):
    @abstractmethod
    def add_data(self):
        pass

    @abstractmethod
    def search(self):
        pass


class LocalBackend(DataBackend):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.dataset.add_faiss_index('embedding')
        
    def add_data(self, dataset: Dataset) -> None:
        self.dataset = concatenate_datasets([self.dataset, dataset])
        self.dataset.add_faiss_index('embedding')

    def search(self, embedding: list[list[float]], k: int, verbose: bool) -> DataFrame:
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
    def dataset_upsert(self, dataset: Dataset, upsert_minibatch_size: int = 1000, text_size: int = 100) -> None:
        assert text_size < 20000, 'text_size should be less then 20000'
        dataset_size = len(dataset)
        for batch_start in range(0, dataset_size, self.upsert_batch_size):
            batch_end = min(batch_start + self.upsert_batch_size, dataset_size)
            data = dataset[batch_start: batch_end] 
            ids = [str(hash(url)) for url in data['url']]
            embeddings = data['embedding']
            metadatas = [{'url': row[0], 'text': row[1][:text_size]} for row in zip(data['url'], data['text'])]
            to_upsert = list(zip(ids, embeddings, metadatas))
            self.index.upsert(vectors=to_upsert, batch_size=upsert_minibatch_size)

    def __init__(self, dataset: Optional(Dataset) = None, api_key: str = None, environment: str ='gcp-starter', index_name: str = None, metric: str = 'euclidean', upsert_batch_size: int = 50000) -> None:
        self.api_key = api_key
        self.environment = environment
        self.upsert_batch_size = upsert_batch_size
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            dimension = len(dataset[0]['embedding'])
            shards = math.ceil(dataset.size_in_bytes / 1024 ** 3)
            pinecone.create_index(index_name, dimension=dimension, shards=shards, metric=metric)
        self.index = pinecone.Index(index_name)
        if dataset is not None:
            self.dataset_upsert(dataset)
        
    def add_data(self, dataset: Dataset) -> None:
        self.dataset_upsert(dataset)

    def search(self, embedding: list[list[float]], k: int, verbose: bool) -> DataFrame:
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
    

class ChromaBackend(DataBackend):
    def dataset_upsert(self, dataset: Dataset) -> None:
        dataset_size = len(dataset)
        for batch_start in range(0, dataset_size, self.upsert_batch_size):
            batch_end = min(batch_start + self.upsert_batch_size, dataset_size)
            data = dataset[batch_start: batch_end] 
            ids = [str(hash(url)) for url in data['url']]
            embeddings = data['embedding']
            documents = data['text']
            metadatas = [{'url': row} for row in data['url']]
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        
    def __init__(self, dataset: Optional[Dataset] = None, type: str = 'ephemeral', collection_name: str = None, upsert_batch_size: int = 5461, **kwargs) -> None:
        if type == 'ephemeral':
            client = chromadb.EphemeralClient(**kwargs)
        if type == 'persistent':
            client = chromadb.PersistentClient(**kwargs)
        if type == 'http':
            client = chromadb.HttpClient(**kwargs)
        self.upsert_batch_size = upsert_batch_size
        self.collection = client.get_or_create_collection(collection_name)
        if dataset is not None:
            self.dataset_upsert(dataset)

    def add_data(self, dataset: Dataset) -> None:
        self.dataset_upsert(dataset)

    def search(self, embedding: list[list[float]], k: int, verbose: bool) -> DataFrame:
        embedding = embedding.tolist()
        answer = self.collection.query(embedding, n_results=k)
        results_dict = {
            'score':[row for row in answer['distances'][0]],
            'text': [row for row in answer['documents'][0]],
            'url': [row['url'] for row in answer['metadatas'][0]],
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