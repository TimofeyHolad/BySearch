from typing import Iterable, Optional
from datasets import load_from_disk, Dataset
import torch
import onnxruntime as ort
from pandas import DataFrame

from .backends import DataBackend, LocalBackend, PineconeBackend, ChromaBackend
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
        pipeline: EmbeddingsPipeline,
        backend: DataBackend, 
        dataset: Optional[Dataset | DataFrame] = None, 
        compute_embeddings: bool = True, 
    ) -> None:
        self.pipeline = pipeline
        self.backend = backend
        # Load and/or preprocess dataset
        if dataset is not None:
            dataset = self.load_dataset(dataset, compute_embeddings=compute_embeddings)
            self.backend.upsert(dataset)

    def upsert(self, dataset: Dataset | DataFrame = None, compute_embeddings: bool = True) -> None:
        dataset = self.load_dataset(dataset, compute_embeddings=compute_embeddings)
        self.backend.upsert(dataset)

    def delete(self, ids: Iterable) -> None:
        self.backend.delete(ids)

    def search(self, prompt: str, k: int = 5, verbose: bool = True) -> DataFrame:
        embedding = self.pipeline([prompt])
        return self.backend.search(embedding, k, verbose)