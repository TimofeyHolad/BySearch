from typing import Optional
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import onnxruntime as ort
from pandas import DataFrame

from .backends import LocalBackend, PineconBackend, ChromaBackend


class BySearch:
    def get_embeddings(self, text_list: list[str]) -> NDArray[np.float64]:
        encoded_input = self.tokenizer(
            text_list, padding='max_length', truncation=True, return_tensors="np", max_length=64, return_token_type_ids=False,
        )
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        model_output = self.session.run(None, input_feed=dict(encoded_input))
        return model_output[0][:, 0]
    
    def load_dataset(self, dataset: Optional[Dataset] = None, path: Optional[str] = None, compute_embeddings: bool = False, batch_size: int = 2) -> Dataset:
        if path is not None:
            dataset = load_from_disk(path)
        if compute_embeddings:
            dataset = dataset.map(
                lambda x: {"embedding": self.get_embeddings(x["text"])}, 
                batched=True,
                batch_size=batch_size, 
            )
        return dataset

    def __init__(self, dataset: Optional[Dataset] = None, path: Optional[str] = None, compute_embeddings: bool = False, tokenizer_checkpoint: str = "KoichiYasuoka/roberta-small-belarusian", model_path: str = 'onnx\\by-model.onnx', backend: str = 'local', **kwargs) -> None:
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.dataset = self.load_dataset(dataset, path, compute_embeddings)
        if backend == 'local':
            self.backend = LocalBackend(self.dataset)
        if backend == 'pinecone':
            self.backend = PineconBackend(self.dataset, **kwargs)
        if backend == 'chroma':
            self.backend = ChromaBackend(self.dataset, **kwargs)

    def add_data(self, dataset: Optional[Dataset] = None, path: str = None, compute_embeddings: bool = False) -> None:
        dataset = self.load_dataset(dataset, path, compute_embeddings)
        self.backend.add_data(dataset)

    def search(self, prompt: str, k: int = 5, verbose: bool = True) -> DataFrame:
        embedding = self.get_embeddings([prompt])
        return self.backend.search(embedding, k, verbose)