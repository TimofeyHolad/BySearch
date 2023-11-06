# BySearch - semantic search framework

BySearch is a simple semantic search package that combine power of open source models with modern vector databases.

My goal is to create approachable and simple solution that would help developers, who are not familiar with NLP models and technologies, to easily build and implement into their projects their mother tongue semantic search engines.  

## Get Started:

```python
import pandas as pd
from bysearch import Engine
from bysearch.pipelines import HuggingFacePipeline, ONNXPipeline
from bysearch.backends import DatasetBackend, PineconeBackend, ChromaBackend 

dataset = pd.DataFrame()
dataset = dataset.add_column('id', list(range(len(dataset))))
backend = ChromaBackend(
    text_column_name='your_text_column_name', 
    id_column_name='your_id_column_name', 
    type='persistent', 
    collection_name='your_collection_name'
)
pipeline = ONNXPipeline.from_hugging_face(
    model='Hugging Face model path', 
    onnx_save_path='local', 
    max_context_length=127,
    verbose=True,
)
engine = Engine(dataset=dataset, pipeline=pipeline, backend=backend)
result = engine.search('аповесць беларускага пісьменніка Уладзіміра Караткевіча', verbose=False)
print(result)
print()
```
