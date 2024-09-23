from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
import re
import ast
import hashlib
from typing import Dict, List


from chatie.formatter import Formatter
from model.embed import Embed
from chatie.config import *
from chatie.milvus import Milvus
from chatie.classifier import Classifier
from utils.logs import ChatIELogger

class CustomExtractor:
    def __init__(self, llm:BaseLanguageModel, embedding:Embed, templates:dict[str:str]):
        self.llm = llm
        self.embedding = embedding
        self.templates = templates
        # initialization
        self.params = self.load_params()  # list of tuple like (name, symbol, unit)
        # handlers
        self.milvus = Milvus()
        self.classifier = Classifier(self.params, self.llm, self.embedding, self.templates)
        self.formatter = Formatter(self.params)
        self.logger = ChatIELogger()

    # ----------------------functions for processing-----------------------
    # initialize all params
    def load_params(self, params_dir:str = PARAMS_INFO_PATH):
        self.logger.info(f'loading params from given params knowledge...')
        with open(params_dir, 'r') as f:
            lines = f.readlines()
        params = []
        for line in lines:
            if line.strip() == '' or line.startswith('#'):
                continue
            symbol, name, unit = [part.strip() for part in line.split(': ')][:3]
            params.append((name, symbol, unit))
        return params
    

    # ----------------------functions for formatter-----------------------
    def filter_phrase(self, input_text:str):
        # filter phrases based on input text
        self.logger.info('filtering phrases...')
        return self.formatter.filter(input_text)
    

    # ----------------------functions for milvus-----------------------
    # upload examples
    def upload(self, examples_path:str=EXAMPLES_PATH):
        self.logger.info('uploading shots...')
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
        self.logger.info('retrieving shots...')
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

    
    # ----------------------functions for classifier-----------------------
    # train classifier net    
    def train_classifier(self):
        self.logger.info('traning classifier net...')
        self.classifier.train_net()

    # ----------------------main functions-----------------------
    # preprocess natural language input
    def extract_params_preprocess(self, query:str, examples:List[Dict])->str:
        self.logger.info('preprocessing query, reforming input...')
        # define templates
        example_prompt = PromptTemplate.from_template(self.templates['params_extraction_example_prompt'])
        suffix = self.templates['params_extraction_suffix_prompt']
        prefix = self.templates['params_extraction_preprocess_prompt']
        prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, suffix=suffix, prefix=prefix, input_variables=['input_text'])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
        r = chain(inputs={'input_text':query})['text']
        return r
    
    # extract all params from preprocessed input
    def extract_params_main(self, query:str, examples:List[Dict]):
        self.logger.info('processing query, extracting params&values&units...')
        # define templates
        example_prompt = PromptTemplate.from_template(self.templates['params_extraction_example_prompt'])
        suffix = self.templates['params_extraction_suffix_prompt']
        prefix = self.templates['params_extraction_prompt']
        prompt = FewShotPromptTemplate(examples=examples, example_prompt=example_prompt, suffix=suffix, prefix=prefix, input_variables=['input_text', 'knowledge'])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
        # prepare knowledge
        phrases = self.formatter.filter(query)
        knowledge = '，'.join(phrases)
        r = chain(inputs={'input_text':query, 'knowledge':knowledge})['text']
        pattern = r'\([^\)]+\)\s*'
        matches = re.findall(pattern, r)
        extraction = []
        for match in matches:
            name, value, unit = ast.literal_eval(match.strip())
            extraction.append((name, value, unit))  # list of tuple like (name, value, unit)
        return extraction

    # reinforce result
    def extract_params_postprocess(self, extraction:List[tuple])->tuple:
        self.logger.info('postprocessing query, re-classifying phrases and units...')
        param_names = [name for name, _, _ in self.params]
        fixed_extraction = []
        # obtain correct name&symbol
        for name, value, unit in extraction:
            if name not in param_names: 
                fixed_name = self.classifier.classify(name)
                self.logger.info(f'fix {name} to {fixed_name} by classifier')
            else: fixed_name = name
            fixed_symbol, standard_unit = [(symbol, unit) for name, symbol, unit in self.params if name == fixed_name][0]
            fixed_extraction.append((fixed_name, fixed_symbol, value, unit, standard_unit))  # list of tuple like (name, symbol, value, unit, standard_unit)
        # handle unit difference
        eqs, ineqs = self.formatter.handle(fixed_extraction)
        return eqs, ineqs
        


    
    def extract(self, query:str):
        # retrieve&process examples for database
        examples = self.retrieve(query)
        examples4preprocess = [{'example_question':example['question_r'], 'example_answer':example['question_p']} for example in examples]
        examples4extract = [{'example_question':example['question_p'], 'example_answer':example['answer']} for example in examples]
        # extract params
        query = self.extract_params_preprocess(query, examples4preprocess)
        self.logger.info(f'extract preprocess result:{query}')
        extraction = self.extract_params_main(query, examples4extract)
        self.logger.info(f'extract process result:{extraction}')
        eqs, ineqs = self.extract_params_postprocess(extraction)
        # write into feature files
        with open(INPUT_CASE_PATH, 'w') as f:
            f.write('\n'.join(eqs))
        with open(INEQS_PATH, 'w') as f:
            f.write('\n'.join(ineqs))

