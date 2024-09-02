import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

config = load_config()

OPENAI_API_KEY = config['OPENAI']['API_KEY']
GPT35 = config['OPENAI']['GPT35']
GPT4 = config['OPENAI']['GPT4']
GPT4O = config['OPENAI']['GPT4O']

OPENAI_EMBEDDING = config['EMBEDDING']['OPENAI_EMBEDDING']

GROQ_API_KEY = config['GROQ']['API_KEY']
GROQ_LLAMA3_8B = config['GROQ']['LLAMA3_8B']
GROQ_LLAMA3_70B = config['GROQ']['LLAMA3_70B']
GROQ_MIXTRAL_8x7B = config['GROQ']['MIXTRAL_8x7B']
GROQ_GEMMA_7B = config['GROQ']['GEMMA_7B']

MONGODB_DB_NAME = config['MONGODB']['DB_NAME']
MONGODB_COLLECTION_NAME = config['MONGODB']['COLLECTION_NAME']
MONGODB_BASE_DIRECTOR = config['MONGODB']['BASE_DIRECTOR']
MONGODB_HOST = config['MONGODB']['HOST']
MONGODB_PORT = int(config['MONGODB']['PORT'])
MONGODB_FEATURES = ["个人史", "过敏史", "婚育史", "家族史", "体格检查", "诊疗经过", "主诉", "现病史", "既往史"]
