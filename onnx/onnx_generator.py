import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime import InferenceSession

# load model and tokenizer
checkpoint = 'KoichiYasuoka/roberta-small-belarusian'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# get input for models and torch model output
input = tokenizer(
    ['аповесць беларускага пісьменніка Уладзіміра Караткевіча', 'аповесць беларускага пісьменніка Уладзіміра Караткевіча'], 
    return_tensors='pt', 
    return_token_type_ids=False,
    padding='max_length',
    max_length=64
)
torch_output = model(**input)

# export onnx model
torch.onnx.export(
    model=model,
    args=tuple(input.values()),
    f='onnx\\by-model.onnx',
    input_names=list(input.keys()),
    output_names=list(torch_output.keys()),
    dynamic_axes={
        **{k: {0: 'batch_size', 1: 'sequence'} for k in list(input.keys())},
        **{k: {0: 'batch_size', 1: 'sequence'} for k in list(torch_output.keys())},
    },
    do_constant_folding=True,
    opset_version=13,
)

# inference of onnx model
session = InferenceSession('onnx\\by-model.onnx', providers=['CPUExecutionProvider'])
onnx_output = session.run(
    None,
    {k: v.numpy() for k, v in input.items()},
)

# check for the size of onnx output error 
error = np.abs(np.max(torch_output[0].detach().numpy() - onnx_output[0]))
print(f'Error: {error}')