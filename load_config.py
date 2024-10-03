import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config.dev.ini')
    return config

config = load_config()

OPENAI_API_KEY = config['OPENAI']['API_KEY']
GPT35 = config['OPENAI']['GPT35']
GPT4 = config['OPENAI']['GPT4']
GPT4O = config['OPENAI']['GPT4O']
OPENAI_EMBEDDING_MODEL = config['OPENAI']['EMBEDDING']

ALI_API_KEY = config['ALI']['API_KEY']
ALI_BASE_URL = config['ALI']['BASE_URL']
QWEN_PLUS = config['ALI']['QWEN_PLUS']
ALI_EMBEDDING_MODEL = config['ALI']['EMBEDDING']

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

MEM0_API_KEY = config['MEM0']['API_KEY']
