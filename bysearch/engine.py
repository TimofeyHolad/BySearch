from typing import Iterable
from datasets import load_from_disk, Dataset
import torch
import onnxruntime as ort
from pandas import DataFrame

from .backends import LocalBackend, PineconeBackend, ChromaBackend
from .pipelines import EmbeddingsPipeline, ONNXPipeline

class BySearch:
    def load_dataset(
        self, 
        dataset: Dataset | DataFrame = None, 
        compute_embeddings: bool = True, 
        batch_size: int = 2
    ) -> Dataset:
        # Convert dataset to HuggingFace Dataset if it is pandas DataFrame 
        try: 
            dataset = Dataset.from_pandas(dataset)
        except:
            pass
        # Compute embeddings from text column if necessary 
        if compute_embeddings:
            dataset = dataset.map(
                lambda x: {"embedding": self.pipeline(x[self.text_column_name])}, 
                batched=True,
                batch_size=batch_size,
            )
        return dataset

    def __init__(
        self, 
        dataset: Dataset | DataFrame = None, 
        text_column_name: str = None, 
        id_column_name: str = None, 
        compute_embeddings: bool = True, 
        pipeline: EmbeddingsPipeline = None,
        backend: str = 'local', 
        **kwargs
    ) -> None:
        
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.pipeline = pipeline
        # Load and/or preprocess dataset
        dataset = self.load_dataset(dataset, compute_embeddings=compute_embeddings)
        # Create backend for required database
        # TODO pass DataBackend argumentz
        if backend == 'local':
            self.backend = LocalBackend(dataset, text_column_name, id_column_name)
        elif backend == 'pinecone':
            self.backend = PineconeBackend(dataset, text_column_name, id_column_name, **kwargs)
        elif backend == 'chroma':
            self.backend = ChromaBackend(dataset, text_column_name, id_column_name, **kwargs)

    def upsert(self, dataset: Dataset | DataFrame = None, compute_embeddings: bool = True) -> None:
        dataset = self.load_dataset(dataset, compute_embeddings=compute_embeddings)
        self.backend.upsert(dataset)

    def delete(self, ids: Iterable) -> None:
        self.backend.delete(ids)

    def search(self, prompt: str, k: int = 5, verbose: bool = True) -> DataFrame:
        embedding = self.pipeline([prompt])
        return self.backend.search(embedding, k, verbose)