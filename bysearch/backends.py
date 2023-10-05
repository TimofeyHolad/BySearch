from typing import Optional, Iterable
import math
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame
from datasets import concatenate_datasets, Dataset
import pinecone
import chromadb


def print_dataframe(df: DataFrame) -> None:
    for _, row in df.iterrows():
        print(144 * '-')
        for column in df.columns:
            print('{}: {}'.format(column, row[column]))


def preprocess_column_names(column_names: list[str], text_column_name: str, id_column_name: str, drop_text_column_name: bool = False, drop_embedding_column_name: bool = True, drop_id_column_name: bool = False) -> list[str]:
    column_names = column_names.copy()
    column_names.remove('embedding')
    column_names.remove(text_column_name)
    column_names.remove(id_column_name)
    if not drop_text_column_name:
        column_names.insert(0, text_column_name)
    if not drop_embedding_column_name:
        column_names.insert(0, 'embedding')
    if not drop_id_column_name:
        column_names.insert(0, id_column_name)
    return column_names


class DataBackend(ABC):
    @abstractmethod
    def upsert(self):
        pass

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def delete(self):
        pass


class LocalBackend(DataBackend):
    def __init__(self, dataset: Dataset, text_column_name: str, id_column_name: str) -> None:
        self.dataset = dataset
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.column_names = preprocess_column_names(dataset.column_names, text_column_name, id_column_name)
        self.dataset.add_faiss_index('embedding')
        
    def upsert(self, dataset: Dataset) -> None:
        self.dataset = concatenate_datasets([self.dataset, dataset])
        self.dataset.add_faiss_index('embedding')
    
    def delete(self) -> None:
        return None

    def search(self, embedding: NDArray[np.float64], k: int, verbose: bool) -> DataFrame:
        scores, samples = self.dataset.get_nearest_examples('embedding', embedding, k=k)
        results_df = DataFrame.from_dict(samples)
        results_df.insert(0, 'score', scores)
        results_df = results_df.reindex(columns=(['score'] + self.column_names))
        if verbose:
            print_dataframe(results_df)
        return results_df
    

class PineconeBackend(DataBackend):
    def dataset_upsert(self, dataset: Dataset, upsert_minibatch_size: int = 1000, text_size: int = 100) -> None:
        # Check for text_size.
        # Pinecone has limitation for text size in metadata.
        assert text_size < 20000, 'text_size should be less then 20000'
        # Prepare column_names list in the way: [text_column_name, *metadata_column_names_without_embeddings]
        column_names = preprocess_column_names(dataset.column_names, self.text_column_name, self.id_column_name, drop_id_column_name=True)
        dataset_size = len(dataset)
        for batch_start in range(0, dataset_size, self.upsert_batch_size):
            batch_end = min(batch_start + self.upsert_batch_size, dataset_size)
            data = dataset[batch_start: batch_end] 
            ids = [str(id)[:512] for id in data[self.id_column_name]]
            embeddings = data['embedding']
            metadata_list = [data[column_name] for column_name in column_names]
            metadata = [{column_names[i]: (row[i][:text_size] if column_names[i] == self.text_column_name else row[i]) for i in range(len(row))} for row in zip(*metadata_list)]
            to_upsert = list(zip(ids, embeddings, metadata))
            self.index.upsert(vectors=to_upsert, batch_size=upsert_minibatch_size)

    def __init__(self, dataset: Optional[Dataset] = None, text_column_name: str = None, id_column_name: str = None, upsert_batch_size: int = 50000, index_name: str = None, metric: str = 'euclidean', **kwargs) -> None:
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.upsert_batch_size = upsert_batch_size
        pinecone.init(**kwargs)
        if index_name not in pinecone.list_indexes():
            dimension = len(dataset[0]['embedding'])
            shards = math.ceil(dataset.size_in_bytes / 1024 ** 3)
            pinecone.create_index(index_name, dimension=dimension, shards=shards, metric=metric)
        self.index = pinecone.Index(index_name)
        if dataset is not None:
            self.dataset_upsert(dataset)
        
    def upsert(self, dataset: Dataset) -> None:
        self.dataset_upsert(dataset)
    
    def delete(self, ids: Iterable) -> None:
        self.index.delete(ids)

    def search(self, embedding: NDArray[np.float64], k: int, verbose: bool) -> DataFrame:
        embedding = embedding.tolist()
        answer = self.index.query(embedding, top_k=k, include_values=False, include_metadata=True)
        scores = [row['score'] for row in answer['matches']]
        ids = [row['id'] for row in answer['matches']]
        samples = [row['metadata'] for row in answer['matches']]
        results_df = DataFrame.from_dict(samples)
        results_df.insert(0, self.id_column_name, ids)
        results_df.insert(0, 'score', scores)
        if verbose:
            print_dataframe(results_df)
        return results_df
    

class ChromaBackend(DataBackend):
    def dataset_upsert(self, dataset: Dataset) -> None:
        column_names = preprocess_column_names(dataset.column_names, self.text_column_name, self.id_column_name, drop_id_column_name=True, drop_text_column_name=True)
        dataset_size = len(dataset)
        for batch_start in range(0, dataset_size, self.upsert_batch_size):
            batch_end = min(batch_start + self.upsert_batch_size, dataset_size)
            data = dataset[batch_start: batch_end] 
            ids = [str(id) for id in data[self.id_column_name]] #?
            embeddings = data['embedding']
            documents = data[self.text_column_name]
            data_list = [data[column_name] for column_name in column_names]
            metadata = [{column_names[i]: value for i, value in enumerate(row)} for row in zip(*data_list)]
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadata)
        
    def __init__(self, dataset: Optional[Dataset] = None, text_column_name: str = None, id_column_name: str = None, upsert_batch_size: int = 5461, type: str = 'ephemeral', collection_name: str = None, **kwargs) -> None:
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.upsert_batch_size = upsert_batch_size
        if type == 'ephemeral':
            client = chromadb.EphemeralClient(**kwargs)
        if type == 'persistent':
            client = chromadb.PersistentClient(**kwargs)
        if type == 'http':
            client = chromadb.HttpClient(**kwargs)
        self.collection = client.get_or_create_collection(collection_name)
        if dataset is not None:
            self.dataset_upsert(dataset)

    def upsert(self, dataset: Dataset) -> None:
        self.dataset_upsert(dataset)

    def delete(self, ids: Iterable) -> None:
        self.collection.delete(ids)

    def search(self, embedding: NDArray[np.float64], k: int, verbose: bool) -> DataFrame:
        embedding = embedding.tolist()
        answer = self.collection.query(embedding, n_results=k)
        scores = answer['distances'][0]
        ids = answer['ids'][0]
        texts = answer['documents'][0]
        samples = answer['metadatas'][0]
        results_df = DataFrame.from_dict(samples)
        results_df.insert(0, self.text_column_name, texts)
        results_df.insert(0, self.id_column_name, ids)
        results_df.insert(0, 'score', scores)
        if verbose:
            print_dataframe(results_df)
        return results_df