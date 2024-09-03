import re
import os
from typing import List, Tuple
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
import hashlib
from utils.logs import CustomLogger

# 定义不同文件的分割符，取决于文件的处理方法
from rag.custom.config import PDF_SEPARATOR

class FileLoader:
    def __init__(self):
        pass
    
    # 对所有的读取进行基本的处理，移除\s以及一些Unicode特殊字符范围
    def base(self, text:str)->str:
        text = re.sub(r'[\uf000-\uf0ff\s\ufff0-\uffff]+', '', text)
        return text
        

    # 封装所有的字符串预处理操作
    def process(self, docs:List[Document])->List[Document]:
        for doc in docs:
            s = doc.page_content
            s = self.base(s)
            doc.page_content = s

    # 处理pdf
    def load_pdf(self, file_path):
        loader = PDFMinerLoader(file_path)
        docs = loader.load()
        self.process(docs)
        return docs
        



class FileSplitter:
    def __init__(self, chunk_size, chunk_overlap, chunk_limit):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_limit = chunk_limit
        self.logger = CustomLogger()
    
    # 按照指定字符进行分割
    def split_by_character(self, docs:List[Document], separator:str)->List[Document]:
        splitter = CharacterTextSplitter(separator, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = splitter.split_documents(docs)
        l1 = len(docs)
        docs = [doc for doc in docs if len(doc.page_content) < self.chunk_limit]
        l2 = len(docs)
        self.logger.info(f'RAG:chunk abolish {l1 - l2} doc(s) because size exceeded the limitation')
        return docs






class FileHandler:
    def __init__(self, chunk_size:int, chunk_overlap:int, chunk_limit:int):
        self.loader = FileLoader()
        self.splitter = FileSplitter(chunk_size, chunk_overlap, chunk_limit)
    
    def handle(self, file_path:str):
        file_name = os.path.basename(file_path)
        chunk_ids = []
        # 处理pdf文件
        if file_path.lower().endswith('.pdf'):
            docs = self.loader.load_pdf(file_path)
            docs = self.splitter.split_by_character(docs, PDF_SEPARATOR)
            for doc in docs:
                chunk_id = hashlib.sha256(doc.page_content.encode('utf-8')).hexdigest()[:32]  # 以内容为seed生成sha256哈希标志，保证相同的内容不会重复上传
                chunk_ids.append(chunk_id)
        elif file_path.lower().endswith('.docx'):
            pass
        # 返回分割后的docs，文件名称以及和docs对应的每一个doc的chunk_id
        return docs, file_name, chunk_ids
            




