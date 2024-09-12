from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
import re
import ast
import hashlib
from typing import Dict, List
import pickle
import matplotlib.pyplot as plt


from model.embed import Embed
from chatie.config import *
from chatie.milvus import Milvus
from chatie.classifier import Classifier

class CustomExtractor:
    def __init__(self, llm:BaseLanguageModel, embedding:Embed, templates:dict[str:str]):
        # init
        self.llm = llm
        self.embedding = embedding
        self.templates = templates
        # params
        self.params = []  # list of tuple like (name, symbol)
        self.extraction = []  # list of tuple like (name, value, unit)
        # handlers
        self.milvus = Milvus()
        self.classifier = Classifier(self.embedding, self.templates)

    # initialize all params
    def load_params(self, params_dir:str = PARAMS_INFO_PATH):
        with open(params_dir, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.strip() == '' or line.startswith('#'):
                continue
            symbol, name = [part.strip() for part in line.split(': ')][:2]
            self.params.append((name, symbol))
        self.classifier.set_params(self.params)

    # upload examples
    def upload(self, examples_path:str=EXAMPLES_PATH):
        self.clear_data()
        with open(examples_path, 'r') as f:
            r = f.read()
        datas = ast.literal_eval(r)
        ids, embeddings, question_r, question_p, answer = [], [], [], [], []
        for data in datas:
            id = hashlib.sha256(data['question_r'].encode('utf-8')).hexdigest()[:32]
            ids.append(id)
            embedding = self.embedding.embedding_single(data['question_r'])
            embeddings.append(embedding)
            question_r.append(data['question_r'])
            question_p.append(data['question_p'])
            answer.append(data['answer'])
        self.milvus.insert(ids, embeddings, question_r, question_p, answer)
    
    # retrieve examples
    def retrieve(self, query:str)->List[Dict]:
        query_embedding = self.embedding.embedding([query])
        results = self.milvus.search(query_embedding, top_k=TOP_K)
        return results

    # check example database
    def list_data(self, limit):
        r = self.milvus.query(expr='', limit=limit)
        return r
    def clear_data(self):
        self.milvus.clear()
        self.milvus.init()  # delete and re-initialize

    # auto generate training data
    def generate_data(self, save_path:str=NET_TRAINING_DATA_PATH):
        # use LLM to generate structured training data
        data_list=[]
        for name, symbol in self.params:
            prompt = PromptTemplate.from_template(self.templates['data_generation_prompt'])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            r = chain(inputs={'input_text':name})['text']
            print(r)
            # format should be a python list, because it's an easy task so we can fully trust LLM
            results = ast.literal_eval(r)
            data_list.extend([(result, name) for result in results])
        with open(save_path, 'wb') as f:
            pickle.dump(data_list, f)
    
    # train the classifier net
    def train_net(self):
        self.generate_data()
        with open(NET_TRAINING_DATA_PATH, 'rb') as f:
            arr = pickle.load(f)
        datas = [name_replaced for name_replaced, name in arr]
        labels = [name for name_replaced, name in arr]
        losses = self.classifier.net_train(datas, labels)
        self.classifier.net_save()
    
    # load the classifier net
    def load_net(self):
        self.classifier.net_load()
    
    # use the classifier net
    def classify(self, param:str):
        return self.classifier.net_predict(param)
    
    # preprocess natural language input
    def extract_params_preprocess(self, query:str, examples:List[Dict])->str:
        example_prompt = PromptTemplate.from_template(self.templates['params_extraction_example_prompt'])
        suffix = self.templates['params_extraction_suffix_prompt']
        prefix = self.templates['params_extraction_preprocess_prompt']
        prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, suffix=suffix, prefix=prefix, input_variables=['input_text'])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
        r = chain(inputs={'input_text':query})['text']
        return r
    

    # extract all params from preprocessed input
    def extract_params_main(self, query:str, examples:List[Dict]):
        example_prompt = PromptTemplate.from_template(self.templates['params_extraction_example_prompt'])
        suffix = self.templates['params_extraction_suffix_prompt']
        prefix = self.templates['params_extraction_prompt']
        prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, suffix=suffix, prefix=prefix, input_variables=['input_text', 'knowledge'])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
        knowledge = '，'.join([name for name, symbol in self.params])
        r = chain(inputs={'input_text':query, 'knowledge':knowledge})['text']
        pattern = r'\([^\)]+\)\s*'
        matches = re.findall(pattern, r)
        for match in matches:
            name, value, unit = ast.literal_eval(match.strip())
            self.extraction.append((name, value, unit))  # list of tuple like (name, value, unit)


    # reinforce result
    def extract_params_postprocess(self):
        pass


    # extract prrams
    def extract(self, query:str, train_net:bool=False):
        # retrieve&process examples for database
        examples = self.retrieve(query)
        examples4preprocess = [{'example_question':example['question_r'], 'example_answer':example['question_p']} for example in examples]
        examples4extract = [{'example_question':example['question_p'], 'example_answer':example['answer']} for example in examples]
        # load params&train net/load net
        self.load_params()
        if train_net: self.train_net()
        self.load_net()
        # extract params
        query = self.extract_params_preprocess(query, examples4preprocess)
        print(f'extract preprocess result:{query}')
        self.extract_params_main(query, examples4extract)
        print(f'extract main result:{self.extraction}')
        
