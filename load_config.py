import configparser
import os

def load_config():
    # env = os.environ.get('AGENT_ENV')
    config = configparser.ConfigParser()
    # if not env:
    #     env = 'dev'
    # config.read('config.%s.ini'% env)
    config.read('config.sit.ini')
    return config

config = load_config()

# EMBEDDING_MODEL = config['OLLAMA']['EMBEDDING_MODEL']
# EMBEDDING_DIMENSION = config['OLLAMA']['EMBEDDING_DIMENSION']
# CHAT_MODEL = config['OLLAMA']['CHAT_MODEL']
# API_KEY = config['OLLAMA']['API_KEY']
# HOST = config['OLLAMA']['HOST']

EMBEDDING_MODEL = config['OPENAI']['EMBEDDING_MODEL']
EMBEDDING_DIMENSION = config['OPENAI']['EMBEDDING_DIMENSION']
CHAT_MODEL = config['OPENAI']['CHAT_MODEL']
API_KEY = config['OPENAI']['API_KEY']

NEO4J_URI = config['NEO4J']['NEO4J_URI']
NEO4J_USERNAME = config['NEO4J']['NEO4J_USERNAME']
NEO4J_PASSWORD = config['NEO4J']['NEO4J_PASSWORD']

MONGODB_DB_NAME = config['MONGODB']['DB_NAME']
MONGODB_COLLECTION_NAME = config['MONGODB']['COLLECTION_NAME']
MONGODB_HOST = config['MONGODB']['HOST']
MONGODB_PORT = int(config['MONGODB']['PORT'])
MONGODB_FEATURES = ["个人史", "过敏史", "婚育史", "家族史", "体格检查", "诊疗经过", "主诉", "现病史", "既往史"]
CASE_HISTORY_BASE_DIRECTOR = config['MONGODB']['CASE_HISTORY_BASE_DIRECTOR']

WEB_SOCKET_PORT = config['SOCKET']['PORT']

LOCAL_QWEN32B = config["LOCAL"]["QWEN25_32B"]
LOCAL_DEEPSEEK_R1_32B = config["LOCAL"]["DEEPSEEK_R1_32B"]
LOCAL_DEEPSEEK_R1_70B = config["LOCAL"]["DEEPSEEK_R1_70B"]
LOCAL_BASE_URL = config["LOCAL"]["BASE_URL"]
LOCAL_EMBEDDING_MODEL = ""

# DEEPSEEK_CHAT = config["DEEPSEEK"]["DEEPSEEK_CHAT"]
# DEEPSEEK_REASONER = config["DEEPSEEK"]["DEEPSEEK_REASONER"]
DEEPSEEK_API_KEY = config["DEEPSEEK"]["API_KEY"]
DEEPSEEK_BASE_URL = config["DEEPSEEK"]["BASE_URL"]