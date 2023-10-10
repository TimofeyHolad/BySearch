from typing import Optional, Any, Iterable
import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import onnxruntime as ort
from pandas import DataFrame

from .backends import LocalBackend, PineconeBackend, ChromaBackend


class BySearch:
    def get_embedding(self, text_list: list[str]) -> NDArray[np.float64]:
        max_length = self.tokenizer.model_max_length - 1
        encoded_input = self.tokenizer(
            text_list, 
            padding='max_length', 
            truncation=True, 
            return_tensors="np", 
            max_length=max_length, 
            return_token_type_ids=False,
            return_overflowing_tokens=True,
        )
        # Map each sample of tokens to corresponding input text index 
        sample_to_text = encoded_input.pop('overflow_to_sample_mapping')
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        last_hidden_states, _ = self.session.run(None, input_feed=dict(encoded_input))
        # Get last hidden states grouped by input text 
        sections = np.unique(sample_to_text, return_index=True)[1][1:]
        grouped_last_hidden_states = np.split(last_hidden_states, sections)
        # Aggregate cls tokens in each group
        aggregated_cls_tokens = [np.mean(group[:, 0, :], axis=0) for group in grouped_last_hidden_states]
        aggregated_cls_tokens = np.array(aggregated_cls_tokens)
        return aggregated_cls_tokens

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

    def __init__(self, dataset: Optional[Dataset] = None, path: Optional[str] = None, text_column_name: str = None, id_column_name: str = None, compute_embeddings: bool = False, tokenizer_checkpoint: str = "KoichiYasuoka/roberta-small-belarusian", model_path: str = 'onnx\\model.onnx', backend: str = 'local', **kwargs) -> None:
        self.text_column_name = text_column_name
        self.id_column_name = id_column_name
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        dataset = self.load_dataset(dataset, path, compute_embeddings=compute_embeddings)
        if backend == 'local':
            self.backend = LocalBackend(dataset, text_column_name, id_column_name)
        elif backend == 'pinecone':
            self.backend = PineconeBackend(dataset, text_column_name, id_column_name, **kwargs)
        elif backend == 'chroma':
            self.backend = ChromaBackend(dataset, text_column_name, id_column_name, **kwargs)

    def upsert(self, dataset: Optional[Dataset] = None, path: str = None, compute_embeddings: bool = False) -> None:
        dataset = self.load_dataset(dataset, path, compute_embeddings=compute_embeddings)
        self.backend.upsert(dataset)

    def delete(self, ids: Iterable) -> None:
        self.backend.delete(ids)

    def search(self, prompt: str, k: int = 5, verbose: bool = True) -> DataFrame:
        embedding = self.get_embedding([prompt])
        return self.backend.search(embedding, k, verbose)