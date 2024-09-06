from typing import List
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
import os

from utils.logs import CustomLogger
from rag.custom.file import FileHandler
from rag.custom.transform import Transform
from rag.custom.milvus import Milvus
from rag.custom.config import *


class CustomRAG:
    def __init__(self, llm:BaseLanguageModel, embedding, reranker, templates):
        self.llm = llm
        self.embeddings_handler = embedding
        self.reranker_handler = reranker
        self.file_handler = FileHandler(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, chunk_limit=CHUNK_LIMIT)
        self.transform_handler = Transform(llm, templates=templates)
        self.milvus_handler = Milvus(host=MILVUS_HOST, port=MILVUS_PORT, db_name=MILVUS_DBNAME, collection_name=MILVUS_CLNAME)
        self.logger_handler = CustomLogger()
        
    
    def rag_pre_process(self, file_path:str):
        self.logger_handler.info('RAG:executing RAG pre-process...')
        # 进行文档分析
        file_name = os.path.basename(file_path)
        self.logger_handler.info('RAG:chunking file...')
        docs, file_name, chunk_ids = self.file_handler.handle(file_path)
        # 大模型预处理
        self.logger_handler.info('RAG:use LLM transforming sentences...')
        contents = self.transform_handler.transform_documents(docs)
        # embed
        self.logger_handler.info('RAG:embedding vectors...')
        embeddings = self.embeddings_handler.embedding(contents)
        # 存入milvus数据库
        self.logger_handler.info('RAG:inserting into milvus...')
        self.milvus_handler.insert(file_name=file_name, chunk_ids=chunk_ids, embeddings=embeddings, contents=contents)
        self.logger_handler.info('RAG:pre-process finished successfully.')
        
    
    def rag_post_process(self, query:str, distance_threshold:float=DISTANCE_THRESHOLD, score_threshold:float=SCORE_THRESHOLD, num_threshold:int=NUM_THRESHOLD)->List[str]:
        self.logger_handler.info('RAG:executing RAG post-process...')
        # 对问题进行预处理
        query = self.transform_handler.transform_query(query)
        # 检索向量库相似向量
        self.logger_handler.info('RAG:searching milvus database...')
        query_embedding = self.embeddings_handler.embedding([query])
        results = self.milvus_handler.search(query_embedding, top_k=TOP_K)
        results = [result for result in results if result['distance'] < distance_threshold]  # 根据distance_threshold删除结果中距离过长的向量
        pairs = [[query, result['content']] for result in results]  # 构造reranker输入
        # rerank
        self.logger_handler.info('RAG:reranking results...')
        scores = self.reranker_handler.rerank(pairs)
        r = zip(scores, [result['distance'] for result in results], [result['content'] for result in results])
        r = [(score-distance/DISTANCE_THRESHOLD, content) for score, distance, content in r if score > score_threshold]  # 根据score_threshold删除结果中评分不够的向量
        r.sort(key=lambda x: x[0], reverse=True)  #使用distance和score联合对r进行重排
        r = r[:num_threshold]  # 根据num_threshold保留指定数量的结果，避免输入语言模型的知识数量过多
        r = [content for score, content in r]
        self.logger_handler.info('RAG:post-process finished successfully.')
        return r
    
    def list_data(self, limit):
        r = self.milvus_handler.query(expr='', limit=limit)
        return r

    def clear_data(self):
        self.milvus_handler.clear()
        self.milvus_handler.init()  # 删除之后需要重新调用init生成数据库