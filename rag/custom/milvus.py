from pymilvus import connections, Collection, utility, db, FieldSchema, CollectionSchema, DataType
from typing import List, Any

from rag.custom.config import INDEX_TYPE, INDEX_PARAMS, METRIC_TYPE, SEARCH_PARAMS, MAX_LIMIT


class Milvus:
    def __init__(self, host:str, port:str, db_name:str, collection_name:str):
        connections.connect(host=host, port=port)
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        # 初始化
        self.init()
    
    # 初始化
    def init(self):
        # 检查数据库是否存在
        databases = db.list_database()
        flag = False
        for database in databases:
            if database == self.db_name:
                flag = True
                break
        if flag == False:
            db.create_database(self.db_name)
        db.using_database(self.db_name)
        # 检查表是否存在
        collections = utility.list_collections()
        flag = False
        for collection in collections:
            if collection == self.collection_name:
                flag = True
                self.collection = Collection(self.collection_name)
                break
        if flag == False:
            fields = [
                FieldSchema('chunk_id', dtype=DataType.VARCHAR, max_length=65, is_primary=True),
                FieldSchema('file_name', dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema('embedding', dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema('content', dtype=DataType.VARCHAR, max_length=32768)
            ]
            schema = CollectionSchema(fields, description='collection for custom RAG service')
            self.collection = Collection(self.collection_name, schema)
    
    # 直接插入数据，根据chunk_id进行更新
    def insert(self, file_name:str, chunk_ids:List[str], embeddings:List[List[float]], contents:List[str]):
        datas = []
        for chunk_id, embedding, content in zip(chunk_ids, embeddings, contents):
            data = {}
            data['chunk_id'] = chunk_id
            data['file_name'] = file_name
            data['embedding'] = embedding
            data['content'] = content
            datas.append(data)
        self.collection.upsert(datas)
        index = {
            'index_type': INDEX_TYPE,
            'metric_type': METRIC_TYPE,
            'params': INDEX_PARAMS
        }
        self.collection.create_index('embedding', index)
        self.collection.load()
    
    def clear(self):
        self.collection.drop()

    def query(self, expr:str, limit:int):
        return self.collection.query(expr=expr, output_fields=['chunk_id', 'file_name', 'content'], limit=limit)
    
    def search(self, query_embedding:Any, top_k:int):
        search_params = {
            'metric_type': METRIC_TYPE,
            'params': SEARCH_PARAMS
        }
        r = self.collection.search(data=query_embedding, anns_field='embedding', param=search_params, limit=top_k, output_fields=['content'])
        hits = r[0]
        # 使用列表字典的形式返回结果
        results = []
        for hit in hits:
            d = {}
            d['id'] = hit.id
            d['distance'] = hit.distance
            d['content'] = hit.entity.get('content')
            results.append(d)
        return results
    
    # 判断是否存在某个文件
    def has_file(self, file_name:str):
        r = self.collection.query(expr=f'file_name == \'{file_name}\'', output_fields=['chunk_id', 'content'])
        if len(r) == 0: return False
        else: return True
    
    def list_file(self):
        r = self.collection.query(expr='', output_fields=['file_name'], limit=MAX_LIMIT)
        r = [result['file_name'] for result in r]
        return r
    
    # 用新的一批chunk覆盖原文件
    def cover_file(self, file_name:str, chunk_ids:List[str], embeddings:List[List[float]], contents:List[str]):
        self.collection.delete(expr=f'file_name == \'{file_name}\'')
        self.insert(file_name=file_name, chunk_ids=chunk_ids, embeddings=embeddings, contents=contents)
        
