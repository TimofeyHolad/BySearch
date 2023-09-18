import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk, concatenate_datasets
import onnxruntime as ort


class BySearch:
    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="np", max_length=64,
        )
        encoded_input = {k: v.astype(dtype=np.int64) for k, v in encoded_input.items()}
        model_output = self.session.run(output_names=['last_hidden_state'], input_feed=dict(encoded_input))
        return model_output[0][:, 0]
    
    
    def load_database(self, dataset=None, path=None, compute_embeddings=False, batch_size=1):
        if dataset is not None:
            database = dataset
        elif path is not None:
            database = load_from_disk(path)

        if compute_embeddings:
            database = database.map(
                lambda x: {"embedding": self.get_embeddings(x["text"])}, 
                batched=True,
                batch_size=batch_size, 
            )
        return database


    def __init__(self, dataset=None, path=None, compute_embeddings=False, tokenizer_checkpoint="KoichiYasuoka/roberta-small-belarusian", model_path='C:\\Users\\tihol\\Projects\\PyProjects\\BySearch\\by-model.onnx'):
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        
        self.database = self.load_database(dataset, path, compute_embeddings)
        self.database.add_faiss_index('embedding')
        

    def add_data(self, dataset=None, path=None, compute_embeddings=False):
        additional_database = self.load_database(dataset, path, compute_embeddings)
        self.database = concatenate_datasets([self.database, additional_database])
        self.database.add_faiss_index('embedding')


    def search(self, prompt):
        embedding = self.get_embeddings([prompt])
        scores, samples = self.database.get_nearest_examples('embedding', embedding, k=5)
        results_df = pd.DataFrame.from_dict(samples)
        results_df['scores'] = scores
        results_df.sort_values('scores', ascending=False, inplace=True)
        for _, row in results_df.iterrows():
            print(148 * '-')
            print(f'Scores: {row.scores}')
            print(f'URL: {row.url}')
            print(f'Text: {row.text}')