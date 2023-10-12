from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.typing import NDArray
import torch
import onnxruntime as ort
from transformers import AutoTokenizer


class EmbeddingsPipeline(ABC):
    @abstractmethod
    def __call__(self):
        pass


class ONNXPipeline(EmbeddingsPipeline):
    def __init__(self, tokenizer, onnx_model, max_context_length) -> None:
        # Try to load tokenizer in case tokenizer variable contains tokenizer path in HuggingFace hub
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, )
        except:
            pass
        self.tokenizer = tokenizer
        try:
            onnx_model = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except: 
            pass
        self.onnx_model = onnx_model
        # TODO condition for max_context_length
        self.max_context_length = tokenizer.model_max_length
        self.max_context_length = max_context_length

    def __call__(self, text_list: list[str]) -> NDArray[np.float64]:
        encoded_input = self.tokenizer(
            text_list, 
            padding='max_length', 
            truncation=True, 
            return_tensors="np", 
            max_length=self.max_context_length, 
            return_token_type_ids=False,
            return_overflowing_tokens=True,
        )
        # Get map array from each sample of tokens to corresponding input text index 
        sample_to_text = encoded_input.pop('overflow_to_sample_mapping')
        # Cast each tokenizer output array to np.int64 for ONNX inference
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        last_hidden_states, _ = self.onnx_model.run(None, input_feed=dict(encoded_input))
        # Get last hidden states grouped by input text 
        sections = np.unique(sample_to_text, return_index=True)[1][1:]
        grouped_last_hidden_states = np.split(last_hidden_states, sections)
        # Aggregate CLS tokens in each group
        aggregated_cls_tokens = [np.mean(group[:, 0, :], axis=0) for group in grouped_last_hidden_states]
        aggregated_cls_tokens = np.array(aggregated_cls_tokens)
        return aggregated_cls_tokens
    