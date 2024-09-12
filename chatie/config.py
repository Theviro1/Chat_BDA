PARAMS_INFO_PATH = 'Chat_BDA/config/feature/params.txt'
EXAMPLES_PATH = 'Chat_BDA/config/feature/examples.txt'
# milvus数据库参数
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_DBNAME = 'custom_rag'
MILVUS_CLNAME = 'params_extraction_shots'
# milvus索引参数
INDEX_TYPE='HNSW'  # 构建索引类型
METRIC_TYPE='L2'   # 使用距离类型
INDEX_PARAMS={     # 索引参数
    'M':16,
    'efConstruction':200
}
SEARCH_PARAMS={    # 检索参数
    'ef':100
}
# milvus检索参数
TOP_K = 3

# 神经网络路径
NET_MODEL_WEIGHT_PATH = 'Chat_BDA/chatie/net/model_weight.pth'
NET_TRAINING_DATA_PATH = 'Chat_BDA/chatie/net/data.pkl'
# 神经网络参数
NET_INPUT_DIM = 32
NET_LR = 1e-3
NET_EPOCH = 800
NET_BATCH_SIZE = 32