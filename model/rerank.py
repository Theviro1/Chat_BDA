from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml
import torch
from typing import List
import math
from tqdm import tqdm

class Rerank:
    def __init__(self, reranker_path:str, batch_size:int=20):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model_path = reranker_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
    
    def rerank(self, pairs:List[List[str]])->List[float]:
        scores = []
        # 如果输入的sentences太大batch会占用过多显存报错，需要分割之后迭代处理，用时间换取空间
        for i in tqdm(range(math.ceil(len(pairs) / self.batch_size)), desc=f'reranking batches with size {self.batch_size}'):
            batch_pairs = pairs[i*self.batch_size:(i+1)*self.batch_size]
            inputs = self.tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs, return_dict=True).logits.view(-1,).float()
            batch_scores = torch.sigmoid(outputs).tolist()
            scores.extend(batch_scores)
        return scores
