from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLanguageModel
from typing import Dict, Any, List
from model.model import CustomLLM
import yaml

from rag.custom.rag import CustomRAG

MODEL_CONFIG_DIR = 'Chat_BDA/config/model_config.yaml'
PROMPT_DIR = 'Chat_BDA/config/prompt.yaml'

class Executor:
    def __init__(self, llm: BaseLanguageModel=CustomLLM(MODEL_CONFIG_DIR), config_dir: str=PROMPT_DIR):
        # 读取配置文件
        with open(config_dir, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)
        # 设置参数
        self.llm = llm
        self.rag = CustomRAG(llm=self.llm)
        
    # 意图识别&&任务分类
    def intention_identification(self, input_text:str)->str:
        prompt = PromptTemplate.from_template(self.templates['intention_identification_prompt'])
        llm = self.llm
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        inputs = {'input_text': input_text}
        t = chain(inputs=inputs)['text']
        print(t)
        # 要求r在函数内被处理，是一个字符串对象
        if t=='知识问答':
            r = self.knowledge_base_qa(input_text)
        elif t=='新品设计':
            r = self.product_design()
        elif t=='模型优化':
            r = self.model_optimize()
        else:
            r = {'input_text': input_text, 'text':'无关输入'}
        return r
    # 上传知识文件
    def knowledge_base_update(self, file_path:str):
        self.rag.rag_pre_process(file_path)

    # 知识问答
    def knowledge_base_qa(self, input_text:str)->str:
        # 使用本地LLM（chatglm-4-9b）+ 自定义RAG
        knowledge = '\n'.join(self.rag.rag_post_process(input_text))
        prompt = PromptTemplate.from_template(self.templates['knowledge_base_qa_prompt'])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=True)
        inputs = {'knowledge':knowledge, 'input_text':input_text}
        r = chain(inputs=inputs)['text']
        return r

    # 参数提取
    def params_extraction(self):
        pass
    
    # 冲突解决
    def resolve_conflict(self):
        pass

    # 新品设计
    def product_design(self):
        pass

    # 模型优化
    def model_optimize(self):
        pass
