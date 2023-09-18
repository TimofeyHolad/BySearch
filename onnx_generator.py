import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime import InferenceSession

checkpoint = 'KoichiYasuoka/roberta-small-belarusian'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

input = tokenizer('аповесць беларускага пісьменніка Уладзіміра Караткевіча', return_tensors='pt', return_token_type_ids=False)
torch_output = model(**input)

torch.onnx.export(
    model=model,
    args=tuple(input.values()),
    f='by-model.onnx',
    input_names=list(input.keys()),
    output_names=list(torch_output.keys()),
    do_constant_folding=True,
    opset_version=13,
)

session = InferenceSession('by-model.onnx', providers=['CPUExecutionProvider'])
onnx_output = session.run(
    None,
    {k: v.numpy() for k, v in input.items()},
)
print(np.abs(np.max(torch_output[0].detach().numpy() - onnx_output[0])))