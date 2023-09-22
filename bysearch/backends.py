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
    def __init__(self, dataset, api_key, environment='gcp-starter'):
        self.api_key = api_key
        self.environment = environment
        pinecone.init(api_key=api_key, environment=environment)
        
    def add_data(self, dataset):
        pass

    def search(self, embedding, verbose=True):
        pass