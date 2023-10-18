import numpy as np
import torch
from onnxruntime import InferenceSession

def onnx_exporter(model, tokenizer, onnx_save_path: str, max_context_length: int, dummy_input: list[str], opset_version: int) -> float:
    # Generate Hugging Face model input and model output
    input = tokenizer(
        dummy_input, 
        return_tensors='pt', 
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        max_length=max_context_length,
    )
    torch_output = model(**input)

    # Export ONNX model
    torch.onnx.export(
        model=model,
        args=tuple(input.values()),
        f=onnx_save_path,
        input_names=list(input.keys()),
        output_names=list(torch_output.keys()),
        dynamic_axes={
            **{k: {0: 'batch_size', 1: 'sequence'} for k in list(input.keys())},
            **{k: {0: 'batch_size', 1: 'sequence'} for k in list(torch_output.keys())},
        },
        do_constant_folding=True,
        opset_version=opset_version,
    )

    # ONNX model inference
    session = InferenceSession(onnx_save_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    onnx_output = session.run(
        None,
        {k: v.numpy() for k, v in input.items()},
    )

    # Check the error of ONNX inference
    error = np.abs(np.max(torch_output[0].detach().numpy() - onnx_output[0]))
    return error.item()