from transformers import AutoModel, AutoTokenizer
import yaml
import torch
import math
from typing import List
from tqdm import tqdm

class Embed:
    def __init__(self, embedding_path:str, batch_size:int=20):
        self.device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
        self.model_path = embedding_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
    
    def embedding(self, sentences:List[str])->List[List[float]]:
        embeddings = []
        # 如果输入的sentences太大batch会占用过多显存报错，需要分割之后迭代处理，用时间换取空间
        for i in tqdm(range(math.ceil(len(sentences) / self.batch_size)), desc=f'embedding batches with size {self.batch_size}'):
            batch_sentences = sentences[i*self.batch_size:(i+1)*self.batch_size]
            inputs = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs, return_dict=True)
            batch_embeddings = outputs.last_hidden_state[:, 0]
            batch_embeddings = [batch_embedding.tolist() for batch_embedding in batch_embeddings]
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def embedding_single(self, sentence:str)->List[float]:
        input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(self.device)
        output = self.model(**input, return_dict=True)
        embedding = output.last_hidden_state[:, 0].tolist()[0]
        return embedding

