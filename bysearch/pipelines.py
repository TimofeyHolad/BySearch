from typing import Optional
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.typing import NDArray
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel


def aggregate_enbeddings(last_hidden_states: NDArray[np.float32 | np.float64], last_hidden_state_to_text: NDArray[np.intc]) -> NDArray[np.float32 | np.float64]:
    # Group last hidden states by corresponding texts 
    sections = np.unique(last_hidden_state_to_text, return_index=True)[1][1:]
    grouped_last_hidden_states = np.split(last_hidden_states, sections)
    # Aggregate CLS tokens in each group
    aggregated_cls_tokens = [np.mean(group[:, 0, :], axis=0) for group in grouped_last_hidden_states]
    return np.array(aggregated_cls_tokens)


class EmbeddingsPipeline(ABC):
    @abstractmethod
    def __call__(self):
        pass


class HuggingFacePipeline(EmbeddingsPipeline):
    def __init__(self, tokenizer, model, device, max_context_length: Optional[int] = None) -> None:
        self.device = device
        # Try to load tokenizer in case tokenizer variable contains tokenizer path in HuggingFace hub
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except:
            pass
        self.tokenizer = tokenizer
        # Try to load model in case model variable contains path
        try:
            model = AutoModel.from_pretrained(model)
        except: 
            pass
        self.model = model.to(device)
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            self.max_context_length = tokenizer.model_max_length

    def __call__(self, text_list: list[str]) -> NDArray[np.float64]:
        encoded_input = self.tokenizer(
            text_list, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt", 
            max_length=self.max_context_length, 
            return_token_type_ids=False,
            return_overflowing_tokens=True,
        )
        # Get map array from each sample of tokens to corresponding input text index 
        sample_to_text = encoded_input.pop('overflow_to_sample_mapping').numpy()
        # Inference through Hugging Face model 
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        outputs = self.model(**encoded_input)
        last_hidden_states = outputs['last_hidden_state'].detach().cpu().numpy()
        # Aggregate cls token embeddings of each text 
        aggregated_cls_embeddings = aggregate_enbeddings(last_hidden_states, last_hidden_state_to_text=sample_to_text)
        return aggregated_cls_embeddings

class ONNXPipeline(EmbeddingsPipeline):
    def __init__(self, tokenizer, onnx_model: ort.InferenceSession | str, max_context_length: Optional[int] = None) -> None:
        # Try to load tokenizer in case tokenizer variable contains tokenizer path in HuggingFace hub
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except:
            pass
        self.tokenizer = tokenizer
        # Try to load ONNX session in case onnx_model variable contains path
        try:
            onnx_model = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except: 
            pass
        self.onnx_model = onnx_model
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            self.max_context_length = tokenizer.model_max_length

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
        # Cast each tokenizer output array to np.int64 for ONNX inference and run session
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        last_hidden_states, _ = self.onnx_model.run(None, input_feed=dict(encoded_input))
        # Aggregate cls token embeddings of each text 
        aggregated_cls_embeddings = aggregate_enbeddings(last_hidden_states, last_hidden_state_to_text=sample_to_text)
        return aggregated_cls_embeddings
    