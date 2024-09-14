from transformers import AutoTokenizer, AutoModel
import torch

class Bert:
    def __init__(self, bert_path:str, batch_size:int):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.bert_path = bert_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.bert_path, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
    
    def tokenize(self, text:str):
        tokens = self.tokenizer.tokenize(text)
        return tokens