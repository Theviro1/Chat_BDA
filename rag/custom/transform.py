from typing import List
from langchain_core.documents import Document
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from tqdm import tqdm
import torch

class Transform:
    def __init__(self, llm:BaseLanguageModel, prompt:PromptTemplate):
        self.llm = llm
        self.prompt = prompt
    
    def transform(self, docs:List[Document])->List[str]:
        results = []
        for doc in tqdm(docs, desc='transforming documents'):
            s = doc.page_content
            chain = LLMChain(llm=self.llm, prompt=self.prompt)
            r = chain(inputs={'input_text': s})['text']
            results.append(r)
        return results

