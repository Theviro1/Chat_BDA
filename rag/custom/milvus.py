from pymilvus import connections, Collection, utility, db, FieldSchema, CollectionSchema, DataType
from typing import List, Any

INDEX_TYPE='IVF_FLAT'
METRIC_TYPE='L2'
INDEX_PARAMS={'nlist':20}
SEARCH_PARAMS={'nprobe':20}

class Milvus:
    def __init__(self, host:str, port:str, db_name:str, collection_name:str):
        connections.connect(host=host, port=port)
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = None
        # 初始化
        self.init()
    
    def init(self):
        # 检查数据库是否存在
        databases = db.list_database()
        flag = False
        # 如果输入了一个其他数据库的名称需要报错，检查schema？TODO
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
