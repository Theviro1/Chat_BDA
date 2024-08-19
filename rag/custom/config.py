# 文档分割:file.py
CHUNK_SIZE = 400  # 在处理文档对文档进行分割的时候，每一个文本块的最长大小
CHUNK_OVERLAP = 100  # 在处理文档对文档进行分割的时候，相邻文本块之间的最大重叠
CHUNK_LIMIT = 4000  # 文档被分割之后，如果有一些块比chunk_size大了太多说明可能是一些无效信息或者参考文献，如果大于chunk_limit直接移除这些块
# milvus数据库:milvus.py
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_DBNAME = 'custom_rag'
MILVUS_CLNAME = 'knowledge_base_qa'
# 目录配置:rerank.py/embed.py
KNOWLEDGE_BASE_DIR = '/home/hjl/Chat_BDA/config/knowledge'  # 本地知识库目录
MODEL_CONFIG_DIR = '/home/hjl/Chat_BDA/config/model/model_config.yaml'  # 模型配置目录
# 检索参数:milvus.py
TOP_K = 5  # 检索数据库获取最匹配的向量数量
BATCH_SIZE = 10  # embedding和reranker模型的批次最大大小