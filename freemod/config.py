# neo4j数据库配置
NEO4J_URL='bolt://localhost:7687'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='Theviro1'
# 加载文件路径
INPUT_CASE_PATH = 'Chat_BDA/config/feature/in/input_case.txt'          # 输入参数
INPUT_PATH = 'Chat_BDA/config/feature/middle/input.txt'                # 处理后的输入参数
DEFAULT_BOUND_PATH = 'Chat_BDA/config/feature/middle/default_bound.txt'    # 默认范围
BOUND_PATH = 'Chat_BDA/config/feature/middle/bound.txt'                    # 处理后的范围
FORMULAS_PATH = 'Chat_BDA/config/feature/middle/formulas.txt'              # 处理后的公式
FORMULAS_RAW_PATH = 'Chat_BDA/config/feature/in/formulas_raw.txt'      # 原始公式
INEQS_PATH = 'Chat_BDA/config/feature/in/ineqs.txt'                    # 不等式约束范围
OUTPUT_PATH = 'Chat_BDA/config/feature/out/output.txt'                  # 输出
# 修正参数
INF_ZERO = 1e-7     # 满足边界关系
ERR_RANGE = 1e-3    # 允许的误差范围
STD_DEV = 5  # 正态分布随机取默认值时的标准差
RETRY_TIMES = 20  # 数值解方程时如果无解会重新随机选取初值，最多重复的次数，超出这个次数之后认为无解
DEFAULT_LOWER_BOUND = -10000
DEFAULT_UPPER_BOUND = 10000
