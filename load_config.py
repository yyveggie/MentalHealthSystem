import configparser
import os
import logging

def load_config():
    """加载配置文件"""
    config = configparser.ConfigParser()
    
    # 根据环境变量选择配置文件，默认使用 sit 环境
    env = os.environ.get('AGENT_ENV', 'sit')
    config_file = f'config.{env}.ini'
    
    if not os.path.exists(config_file):
        config_file = 'config.sit.ini'  # 回退到默认配置
    
    config.read(config_file)
    return config

# 加载配置
config = load_config()

# OpenAI 配置
try:
    API_KEY = config['OPENAI']['API_KEY']
    BASE_URL = config['OPENAI']['BASE_URL']
    CHAT_MODEL = config['OPENAI']['CHAT_MODEL']
    EMBEDDING_MODEL = config.get('OPENAI', 'EMBEDDING_MODEL', fallback='text-embedding-3-small')
    EMBEDDING_DIMENSION = 1536  # OpenAI text-embedding-3-small 的默认维度
except KeyError as e:
    logging.warning(f"OpenAI配置缺失: {e}")
    API_KEY = None
    BASE_URL = None
    CHAT_MODEL = None
    EMBEDDING_MODEL = None
    EMBEDDING_DIMENSION = 1536

# Neo4j 配置
try:
    NEO4J_URI = config['NEO4J']['NEO4J_URI']
    NEO4J_USERNAME = config['NEO4J']['NEO4J_USERNAME']
    NEO4J_PASSWORD = config['NEO4J']['NEO4J_PASSWORD']
except KeyError as e:
    logging.warning(f"Neo4j配置缺失: {e}")
    NEO4J_URI = None
    NEO4J_USERNAME = None
    NEO4J_PASSWORD = None

# MongoDB 配置
try:
    MONGODB_DB_NAME = config['MONGODB']['DB_NAME']
    MONGODB_COLLECTION_NAME = config['MONGODB']['COLLECTION_NAME']
    MONGODB_HOST = config['MONGODB']['HOST']
    MONGODB_PORT = int(config['MONGODB']['PORT'])
    CASE_HISTORY_BASE_DIRECTOR = config['MONGODB']['CASE_HISTORY_BASE_DIRECTOR']
except KeyError as e:
    logging.warning(f"MongoDB配置缺失: {e}")
    MONGODB_DB_NAME = 'medical_records'
    MONGODB_COLLECTION_NAME = 'raw_data'
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    CASE_HISTORY_BASE_DIRECTOR = './database/case'

# MongoDB 特征列表
MONGODB_FEATURES = ["个人史", "过敏史", "婚育史", "家族史", "体格检查", "诊疗经过", "主诉", "现病史", "既往史"]

# WebSocket 配置
try:
    WEB_SOCKET_PORT = int(config['SOCKET']['PORT'])
except (KeyError, ValueError) as e:
    logging.warning(f"WebSocket端口配置缺失或无效: {e}")
    WEB_SOCKET_PORT = 8763

# 其他可能需要的配置
try:
    # 阿里云配置
    ALI_API_KEY = config.get('ALI', 'API_KEY', fallback=None)
    ALI_BASE_URL = config.get('ALI', 'BASE_URL', fallback=None)
    ALI_CHAT_MODEL = config.get('ALI', 'CHAT_MODEL', fallback=None)
    
    # DeepSeek 配置
    DEEPSEEK_API_KEY = config.get('DEEPSEEK', 'API_KEY', fallback=None)
    DEEPSEEK_BASE_URL = config.get('DEEPSEEK', 'BASE_URL', fallback=None)
    DEEPSEEK_CHAT_MODEL = config.get('DEEPSEEK', 'CHAT_MODEL', fallback=None)
    
    # Local 配置
    LOCAL_BASE_URL = config.get('LOCAL', 'BASE_URL', fallback=None)
    LOCAL_CHAT_MODEL = config.get('LOCAL', 'CHAT_MODEL', fallback=None)
    
except Exception as e:
    logging.warning(f"加载其他配置时出错: {e}")

# 打印配置加载状态（用于调试）
if __name__ == "__main__":
    print("配置加载状态:")
    print(f"API_KEY: {'已设置' if API_KEY else '未设置'}")
    print(f"CHAT_MODEL: {CHAT_MODEL}")
    print(f"MONGODB_HOST: {MONGODB_HOST}:{MONGODB_PORT}")
    print(f"NEO4J_URI: {NEO4J_URI}")
    print(f"WEB_SOCKET_PORT: {WEB_SOCKET_PORT}")
