# BySearch - semantic search framework

BySearch is a simple semantic search package that combine power of open source models with modern vector databases.

My goal is to create approachable and simple solution that would help developers, who are not familiar with NLP models and technologies, to easily build and implement into their projects their mother tongue semantic search engines.  

## Get Started

BySearch combines different language models and different vector storages in a united simple semantic search API.

```python
import pandas as pd
from bysearch import Engine
from bysearch.pipelines import ONNXPipeline
from bysearch.backends import ChromaBackend 

# Open your data as Pandas DataFrame
dataset = pd.DataFrame()
# Create vector storage backend
backend = DatasetBackend(text_column_name='your_text_column_name', id_column_name='your_id_column_name')
# Create text processing pipeline
pipeline = HuggingFacePipeline(model='hugging_face_model_path')
# Create engine using pipeline and backend, upsert your data
engine = Engine(dataset=dataset, pipeline=pipeline, backend=backend)
# Perform search by your prompt
result = engine.search('Your prompt', verbose=False)
print(result)
```
Other complex examples are available [here](https://github.com/tiholad/BySearch/blob/main/demo%20EN.ipynb). 
