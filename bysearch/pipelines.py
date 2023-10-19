from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel

from .utils import onnx_exporter


def aggregate_embeddings(last_hidden_states: NDArray[np.float32 | np.float64], last_hidden_state_to_text: NDArray[np.intc]) -> NDArray[np.float32 | np.float64]:
    '''Group by input texts and aggregate last hidden states of CLS tokens from each sample'''
    # Group last hidden states by corresponding texts 
    sections = np.unique(last_hidden_state_to_text, return_index=True)[1][1:]
    grouped_last_hidden_states = np.split(last_hidden_states, sections)
    # Aggregate CLS token last hidden states in each group
    aggregated_cls_tokens = [np.mean(group[:, 0, :], axis=0) for group in grouped_last_hidden_states]
    return np.array(aggregated_cls_tokens)


class EmbeddingsPipeline(ABC):
    @abstractmethod
    def __call__(self):
        pass


class HuggingFacePipeline(EmbeddingsPipeline):
    def __init__(self, model, tokenizer = None, device: str = 'cuda', max_context_length: Optional[int] = None) -> None:
        self.device = device
        if not tokenizer:
            if not isinstance(model, str):
                raise ValueError("Tokenizer parameter can be not specified only in case model parameter contains Hugging Face path.")
            tokenizer = model
        
        # Try to load tokenizer in case tokenizer variable contains tokenizer HuggingFace hub path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except:
            pass
        self.tokenizer = tokenizer
        # Try to load model in case model variable contains model HuggingFace hub path
        try:
            model = AutoModel.from_pretrained(model)
        except: 
            pass
        self.model = model.to(device)
        if not max_context_length:
            max_context_length = tokenizer.model_max_length
        self.max_context_length = max_context_length

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
        # Aggregate all cls token last hidden states by texts 
        aggregated_cls_embeddings = aggregate_embeddings(last_hidden_states, last_hidden_state_to_text=sample_to_text)
        return aggregated_cls_embeddings

class ONNXPipeline(EmbeddingsPipeline):
    def __init__(self, onnx_model: ort.InferenceSession | str, tokenizer, max_context_length: Optional[int] = None) -> None:
        # Try to load tokenizer in case tokenizer variable contains tokenizer HuggingFace hub path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except:
            pass
        self.tokenizer = tokenizer
        # Try to load ONNX session in case onnx_model variable contains local path
        try:
            onnx_model = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except: 
            pass
        self.onnx_model = onnx_model
        if not max_context_length:
            max_context_length = tokenizer.model_max_length
        self.max_context_length = max_context_length

    @staticmethod
    def from_hugging_face(model, tokenizer = None, onnx_save_path: str = 'onnx.model', dummy_input: str = None, max_context_length: Optional[int] = None, opset_version: int = 13):
        if not tokenizer:
            if not isinstance(model, str):
                raise ValueError("Tokenizer parameter can be not specified only in case model parameter contains Hugging Face path.")
            tokenizer = model
        # Try to load tokenizer in case tokenizer variable contains tokenizer HuggingFace hub path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        except:
            pass
        # Try to load ONNX session in case onnx_model variable contains local path
        try:
            model = AutoModel.from_pretrained(model)
        except: 
            pass
        if not max_context_length:
            max_context_length = tokenizer.model_max_length
        # TODO check for file and forced update
        onnx_exporter(model, tokenizer, onnx_save_path, max_context_length, dummy_input, opset_version)
        onnx_model = ort.InferenceSession(onnx_save_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        return ONNXPipeline(onnx_model, tokenizer, max_context_length)

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
        # Get map array (maps each sample of tokens to corresponding input text index) 
        sample_to_text = encoded_input.pop('overflow_to_sample_mapping')
        # Cast each tokenizer output array to np.int64 for ONNX inference and run session
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        # TODO outputs unpacking 
        last_hidden_states = self.onnx_model.run(None, input_feed=dict(encoded_input))[0]
        # Aggregate all cls token last hidden states by texts 
        aggregated_cls_embeddings = aggregate_embeddings(last_hidden_states, last_hidden_state_to_text=sample_to_text)
        return aggregated_cls_embeddings