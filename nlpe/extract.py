from langchain_core.prompts import PromptTemplate
from langchain.llms.base import BaseLanguageModel

from nlpe.config import *
class CustomExtract:
    def __init__(self, llm:BaseLanguageModel):
        self.llm = llm
        self.param_list = []
    
    def load_params(self, config_dir:str = PARAMS_INFO_PATH):
        with open(config_dir, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.strip() == '' or line.startswith('#'):
                continue
            name, info = [part.strip() for part in line.split(': ')][:2]
            self.param_list.append((name, info))

    def extract(self):
        pass