from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLanguageModel
from typing import Dict, Any, List
from model.model import CustomLLM
import yaml

from rag.custom.rag import CustomRAG

class Executor:
    def __init__(self, llm: BaseLanguageModel=CustomLLM(), config_dir: str='/home/hjl/Chat_BDA/config/prompt/prompt.yaml'):
        # 读取配置文件
        with open(config_dir, 'r') as f:
            self.templates = yaml.safe_load(f)
        # 设置参数
        self.llm = llm
        
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
    
    # 知识问答
    def knowledge_base_qa(self, input_text:str)->str:
        # 使用本地LLM（chatglm-4-9b）+ 自定义RAG
        rag = CustomRAG(llm=self.llm, prompt=self.templates['document_transform_prompt'])
        knowledge = '\n'.join(rag.rag_post_process(input_text))
        prompt = PromptTemplate.from_template(self.templates['knowledge_base_qa_prompt'])
        llm = self.llm
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        inputs = {'knowledge':knowledge, 'input_text':input_text}
        r = chain(inputs=inputs)['text']
        return r
        # # 使用QAnything + Qwen-7B，需要本地部署运行QAnything框架，端口是8777
        # from rag.qanything_connector.qanything import send_request
        # question = input_text
        # r = send_request(question)
        # for segement in r['source_documents']:
        #     print('source:\n'+segement['content']+'\n\n')
        # return r['response']

    # 参数提取
    def params_extraction(self, ):
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
