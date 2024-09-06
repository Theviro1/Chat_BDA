# from core.executor import Executor
# executor = Executor()
# query = '''
# 你是一名自然语言实体识别领域的专家，现在你需要对一段话中涉及电池专业知识的实体进行识别，同时对于其值或者取值范围进行识别。
# 请严格遵守如下规则：
# 1. 对于一个实体和其值，以（实体：值）的元组形式输出，元组之间使用空格隔开
# 2. 不需要纠结单位，如果文字中给出了单位就添加，没有给出就不添加
# 3. 如果值是一个取值范围，使用>、<符号来表示
# 请帮我设计一个高性能电池，堆叠数为2，电池厚度不超过2，正极涂层厚度为20，此外我还希望电池的正极辊压反弹压实密度为0.9'
# '''
# r = executor.llm(query)
# print(r)


# freeMOD测试：success
# from freemod import freeMOD
# freeMOD.run()


# from py2neo import Graph, NodeMatcher, RelationshipMatcher
# graph = Graph('bolt://localhost:7687', auth=('neo4j', 'Theviro1'))
# node_matcher = NodeMatcher(graph)
# r = node_matcher.match('Parameter')
# for res in r:
#     print(res)



# neo4j本地测试：success
# from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
# graph = Graph('bolt://localhost:7687', auth=('neo4j', 'Theviro1'), name='neo4j')
# r = graph.run("SHOW DATABASES")
# print(r)



# 知识问答+自定义RAG联合测试：success
from core.executor import Executor
executor = Executor()
query = '如果我想提高电池在高倍率下的放电容量，我可以选用尺寸更大的团聚体结构么？'
# executor.knowledge_base_update('/home/hjl/Chat_BDA/config/knowledge/方儒卿-非体积平均锂离子电池电极层模型的构建与应用.pdf')
r = executor.intention_identification(query)
# r = executor.llm(query)
print(r)


# 自定义logger测试：success
# from utils.logs import CustomLogger
# logger = CustomLogger()
# logger.warning('This is a test warning!')


# 使用大模型对document预处理效果测试：success
# from model.model import CustomLLM
# from pymilvus import connections, db, utility, Collection
# from langchain.prompts import PromptTemplate
# import yaml
# from rag.custom.rag import CustomRAG
# with open('/home/hjl/Chat_BDA/config/prompt/prompt.yaml','r') as f:
#     templates = yaml.safe_load(f)
# llm = CustomLLM()
# prompt = PromptTemplate.from_template(templates['document_transform_prompt'])
# rag = CustomRAG(llm, prompt)
# query = '哪种孔隙率分布构型有利于提高电池高倍率下的性能？造成不同构型产生不同性能的原因是什么？'
# # rag.rag_pre_process()
# r = rag.rag_post_process(query)
# print(len(rag.list_data(limit=1000)))
# print(r)

# easyocr读pdf测试：failed 流程正确，效果太差，easyocr不太行
# import easyocr
# import fitz
# from tqdm import tqdm
# import numpy as np
# import base64
# f = fitz.open('/home/hjl/Chat_BDA/config/knowledge/王子珩-团聚体堆叠型多孔电极模型构建与应用.pdf')
# reader = easyocr.Reader(['ch_sim', 'en'])
# for i in tqdm(range(f.page_count)):
#     page = f.load_page(i)
#     pix = page.get_pixmap()
#     img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))
#     result = reader.readtext(img)
#     for (bbox, text, score) in result:
#         print(text)



# glm4-9b+QAnything-Qwen-7B意图识别、知识问答联合测试：failed 两张GPU卡内存不足以支撑两个模型共同运行
# from core.executor import Executor
# executor = Executor()
# query = '对于一个在零下 20 摄氏度仍能正常工作的电池，提高耐冷度通常需要提高什么电池属性？'
# r = executor.intention_identification(query)
# print(r)



# QAnything本地部署测试：success
# # 命令行启动QAnything：bash ./run.sh -c local -i 0 -b hf -m Qwen-7B-QAnything -t qwen-7b-qanything
# import requests
# import json
# url = "http://localhost:8777/api/local_doc_qa/list_files"
# headers = {
#     "Content-Type": "application/json"
# }
# data = {
# 	"user_id": "zzp",
# 	"kb_id": "KB0677632d6bdf44a9b379fe386f14bdd7"
# }
# response = requests.post(url, headers=headers, data=json.dumps(data))
# print(response.status_code)
# print(response.text)


# langchain工具loader、splitter测试：success
# import re
# from typing import List
# from langchain_community.document_loaders import UnstructuredPDFLoader, PDFMinerLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_core.documents import Document
# def handle(docs:List[Document])->List[Document]:
#     for doc in docs:
#         s = doc.page_content
#         s = re.sub(r'[\uf000-\uf0ff]', '', s)
#         s = re.sub(r'[\t\s]{1,}', '', s)
#         s = re.sub(r'[\n]{1,}', '。', s)
#         doc.page_content = s
    
# def handle_sub(text: str)->str:
#     pattern1 = r'#{3,}'
#     text = re.sub(pattern1, '', text)
#     pattern2 = r'`{3,}'
#     text = re.sub(pattern2, '', text)
#     return text.strip()
# def handle(docs: Iterable[Document])-> Iterable[Document]:
#     for doc in docs:
#         r = handle_sub(doc.page_content)
#         doc.page_content = r
#     return docs
# loader = PDFMinerLoader('/home/hjl/Chat_BDA/config/knowledge/王子珩-团聚体堆叠型多孔电极模型构建与应用.pdf')
# docs = loader.load()
# handle(docs)
# splitter = CharacterTextSplitter(separator='。')
# r = splitter.split_documents(docs)
# for doc in docs:
#     print(doc)
#     print('\n')


# milvus向量数据库测试：success
# from pymilvus import MilvusClient, connections, db, FieldSchema, Collection, CollectionSchema, DataType, utility
# connections.connect('default', host='localhost', port='19530')
# print(db.list_database())
# db.using_database('test')
# fields = [
#     FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
#     FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=2)
# ]
# schema = CollectionSchema(fields=fields, description='test schema')
# collection = Collection(name='test_collection')
# data = [{'id':1,'embedding':[0.8, 0.9]}]
# collection.upsert(data)
# index_params = {
#     'index_type': 'IVF_FLAT',
#     'metric_type': 'L2',
#     'params':{'nlist':1}
# }
# collection.create_index(field_name='embedding', index_params=index_params)
# collection.load()
# print(collection.query(expr='', limit=10, output_fields=['id', 'embedding']))
# print(utility.list_collections())



# embedding模型测试：success
# from transformers import AutoTokenizer, AutoModel
# import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = AutoTokenizer.from_pretrained('/home/hjl/models/embedding/bce-embedding-base_v1')
# model = AutoModel.from_pretrained('/home/hjl/models/embedding/bce-embedding-base_v1')
# model.to(device).eval()
# inputs = tokenizer(['这是用来测试的语句', '这是用来侧式的语句'], padding=True, truncation=True, return_tensors='pt')
# inputs.to(device)
# outputs = model(**inputs, return_dict=True)
# embeddings = outputs.last_hidden_state[:, 0]
# embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
# print(embeddings)



# reranker模型测试：success
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# model = AutoModelForSequenceClassification.from_pretrained('/home/hjl/models/reranker/bce-reranker-base_v1')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device).eval()
# tokenizer = AutoTokenizer.from_pretrained('/home/hjl/models/reranker/bce-reranker-base_v1')
# inputs = tokenizer([['现在几点了', '现在10点'], ['现在几点了', '现在11点了']], padding=True, truncation=True, return_tensors='pt')
# inputs.to(device)
# scores = model(**inputs, return_dict=True).logits.view(-1,).float()
# scores = torch.sigmoid(scores)
# print(scores)



# langchain自定义Executor测试：success
# from core.executor import Executor
# executor = Executor()
# query = '介绍一下Newman P2D模型的几何结构'
# r = executor.intention_identification(query)
# print(r)



# langchain自定义LLM测试：success
# from model.model import CustomLLM
# model = CustomLLM()
# r = model('''
#     你是一名意图识别专家。你的任务是根据给定的用户输入，判断其最可能的意图，并仅返回以下选项之一：知识问答，模型优化，新品设计，无关输入。
#     请严格遵守以下规则：
#     1. 对于任何你不确定或无法判定的输入，请选择'无关输入'。
#     2. 只返回上述四个选项之一，不要添加任何字。
#     现在用户输入如下：
#     我想知道，我设计了一个电池组，如何选择正极材料来提升性能？
#     请你按照规则作答。
# ''')
# print(r)



# prompt加载测试：success
# import yaml
# from langchain.prompts import PromptTemplate
# path = '/home/hjl/Chat_BDA/config/prompt/prompt.yaml'
# with open(path, 'r') as f:
#     templates = yaml.safe_load(f)
# prompt = PromptTemplate(template=templates['intention_identification_prompt']['template'], input_variables=[templates['intention_identification_prompt']['input_variables']])
# inputs = {'input_text': 'Fuck you'}
# r = prompt.format(**inputs)
# print(r)



# 本地模型测试：success
# from transformers import AutoTokenizer, AutoModel
# import torch
# path = '/home/hjl/llm'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# model = AutoModel.from_pretrained(path, trust_remote_code=True)
# model.to(device).eval()
# query = '''
#     你是一名意图识别专家。你的任务是根据给定的用户输入，判断其最可能的意图，并仅返回以下选项之一：知识问答，模型优化，新品设计，无关输入。
#     请严格遵守以下规则：
#     1. 对于任何你不确定或无法判定的输入，请选择'无关输入'。
#     2. 只返回上述四个选项之一，不要添加任何字。
#     现在用户输入如下：
#     我想知道，我设计了一个电池组，如何选择正极材料来提升性能？
#     请你按照规则作答。
# '''
# inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],                                      
#                                        tokenize=True,
#                                        return_tensors="pt",
#                                        return_dict=True
#                                        ).to(device)
# # inputs = tokenizer([query], return_tensors='pt').to(device)
# outputs = model.generate(**inputs, max_new_tokens=10)
# outputs = outputs[:, inputs['input_ids'].shape[1]:]
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(result)