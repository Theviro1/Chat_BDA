from chatie.config import *

from pymilvus import connections, utility, Collection, db, FieldSchema, CollectionSchema, DataType
from typing import List, Any

class Milvus:
    def __init__(self, host:str=MILVUS_HOST, port:str=MILVUS_PORT, db_name:str=MILVUS_DBNAME, collection_name:str=MILVUS_CLNAME):
        connections.connect(host=host, port=port)
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        # 手动初始化
        self.init()

    def init(self):
        # 检查数据库是否存在
        databases = db.list_database()
        exist = True if self.db_name in databases else False
        if not exist: db.create_database(self.db_name)
        db.using_database(self.db_name)
        # 检查表是否存在
        collections = utility.list_collections()
        exist = True if self.collection_name in collections else False
        if exist: self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema('id', dtype=DataType.VARCHAR, max_length=65, is_primary=True),
                FieldSchema('embedding', dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema('question_r', dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema('question_p', dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema('answer', dtype=DataType.VARCHAR, max_length=4096)
            ]
            schema = CollectionSchema(fields, description='collection for param extraction shots')
            self.collection = Collection(self.collection_name, schema)
    
    def insert(self, ids:List[str], embeddings:List[List[float]], question_r:List[str], question_p:List[str], answer:List[str]):
        datas = []
        for id, embedding, qr, qp, a in zip(ids, embeddings, question_r, question_p, answer):
            data = {}
            data['id'] = id
            data['embedding'] = embedding
            data['question_r'] = qr
            data['question_p'] = qp
            data['answer'] = a
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
        return self.collection.query(expr=expr, output_fields=['id', 'question_r', 'question_p', 'answer'], limit=limit)
    
    def search(self, query_embedding:Any, top_k:int):
        search_params = {
            'metric_type': METRIC_TYPE,
            'params': SEARCH_PARAMS
        }
        r = self.collection.search(data=query_embedding, anns_field='embedding', param=search_params, limit=top_k, output_fields=['question_r', 'question_p', 'answer'])
        hits = r[0]
        # 使用列表字典的形式返回结果
        results = []
        for hit in hits:
            d = {}
            d['id'] = hit.id
            d['distance'] = hit.distance
            d['question_r'] = hit.entity.get('question_r')
            d['question_p'] = hit.entity.get('question_p')
            d['answer'] = hit.entity.get('answer')
            results.append(d)
        return results
