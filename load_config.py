import configparser
import os

def load_config():
    env = os.environ.get('AGENT_ENV')
    config = configparser.ConfigParser()
    if not env:
        env = 'dev'
    config.read('config.%s.ini'% env)
    return config

config = load_config()

HUGGINGFACE_EMBEDDING_MODEL = config['HUGGINGFACE']['EMBEDDING_MODEL']

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
