from typing import List
from langchain_core.documents import Document
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from tqdm import tqdm
import yaml

class Transform:
    def __init__(self, llm:BaseLanguageModel, config_dir:str):
        self.llm = llm
        with open(config_dir, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)
    
    def transform_documents(self, docs:List[Document])->List[str]:
        results = []
        prompt = PromptTemplate.from_template(self.templates['transform_documents_prompt'])
        for doc in tqdm(docs, desc='transforming documents'):
            s = doc.page_content
            chain = LLMChain(llm=self.llm, prompt=prompt)
            r = chain(inputs={'input_text': s})['text']
            results.append(r)
        return results
    
    def transform_query(self, query:str)->str:
        # prompt = PromptTemplate.from_template(self.templates['transform_query_prompt'])
        # chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
        # r = chain(inputs={'input_text':query})['text']
        # r = '关于电池中的' + r + '，' + query
        r = '在电池领域中，' + query
        print(r)
        return r


