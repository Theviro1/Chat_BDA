# 文档分割:file.py
CHUNK_SIZE = 400  # 在处理文档对文档进行分割的时候，每一个文本块的最长大小
CHUNK_OVERLAP = 100  # 在处理文档对文档进行分割的时候，相邻文本块之间的最大重叠
CHUNK_LIMIT = 4000  # 文档被分割之后，如果有一些块比chunk_size大了太多说明可能是一些无效信息或者参考文献，如果大于chunk_limit直接移除这些块
PDF_SEPARATOR = '。'
# milvus数据库:milvus.py
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_DBNAME = 'custom_rag'
MILVUS_CLNAME = 'knowledge_base_qa'
# 检索参数:milvus.py
TOP_K = 30  # 检索数据库获取最匹配的向量数量
DISTANCE_THRESHOLD = 100  # 相似度距离限制
SCORE_THRESHOLD = 0.35  # 重排分数限制
NUM_THRESHOLD = 10  # 检索数量限制

INDEX_TYPE='HNSW'  # 构建索引类型
METRIC_TYPE='L2'   # 使用距离类型
INDEX_PARAMS={     # 索引参数
    'M':16,
    'efConstruction':200
}
SEARCH_PARAMS={    # 检索参数
    'ef':100
}

MAX_LIMIT = 10000  # query数量最大限制