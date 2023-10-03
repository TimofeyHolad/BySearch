from typing import Optional, Any
import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import onnxruntime as ort
from pandas import DataFrame

from .backends import LocalBackend, PineconeBackend, ChromaBackend


class BySearch:
    def get_embedding(self, text_list: list[str]) -> NDArray[np.float64]:
        encoded_input = self.tokenizer(
            text_list, padding='max_length', truncation=True, return_tensors="np", max_length=64, return_token_type_ids=False,
        )
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        model_output = self.session.run(None, input_feed=dict(encoded_input))
        return model_output[0][:, 0]
    
    def map_embedding(self, data_dict: dict[str, list[Any]]) -> dict[str, NDArray[np.float64]]:
        text_list = data_dict['text']
        return {'embedding': self.get_embedding(text_list)}

    def load_dataset(self, dataset: Optional[Dataset] = None, path: Optional[str] = None, compute_embeddings: bool = False, batch_size: int = 2) -> Dataset:
        if path is not None:
            dataset = load_from_disk(path)
        if compute_embeddings:
            dataset = dataset.map(
                lambda x: {"embedding": self.get_embedding(x[self.text_column_name])}, 
                batched=True,
                batch_size=batch_size, 
            )
        return dataset

    def __init__(self, dataset: Optional[Dataset] = None, path: Optional[str] = None, text_column_name: str = None, compute_embeddings: bool = False, tokenizer_checkpoint: str = "KoichiYasuoka/roberta-small-belarusian", model_path: str = 'onnx\\by-model.onnx', backend: str = 'local', **kwargs) -> None:
        self.text_column_name = text_column_name
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        dataset = self.load_dataset(dataset, path, compute_embeddings=compute_embeddings)
        if backend == 'local':
            self.backend = LocalBackend(dataset, text_column_name)
        if backend == 'pinecone':
            self.backend = PineconeBackend(dataset, text_column_name, **kwargs)
        if backend == 'chroma':
            self.backend = ChromaBackend(dataset, text_column_name, **kwargs)

    def add_data(self, dataset: Optional[Dataset] = None, path: str = None, compute_embeddings: bool = False) -> None:
        dataset = self.load_dataset(dataset, path, compute_embeddings=compute_embeddings)
        self.backend.add_data(dataset)

    def search(self, prompt: str, k: int = 5, verbose: bool = True) -> DataFrame:
        embedding = self.get_embedding([prompt])
        return self.backend.search(embedding, k, verbose)