import torch
from transformers import AutoTokenizer, AutoModel
from transformers.onnx import FeaturesManager, export

checkpoint = 'KoichiYasuoka/roberta-small-belarusian'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

input = tokenizer('аповесць беларускага пісьменніка Уладзіміра Караткевіча', return_tensors='pt')
print(tuple(input.values()))

rez = torch.onnx.export(
    model=model,
    args=tuple(input.values()),
    f='by-model.onnx',
    input_names=['input_ids', 'token_type_ids', 'attention_mask'],
    output_names=['last_hidden_state'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                  'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence'},
                  'last_hidden_state': {0: 'batch_size', 1: 'sequence'}},
    do_constant_folding=True,
    opset_version=13,
)
print(rez)