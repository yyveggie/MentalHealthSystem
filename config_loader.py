import configparser
import inspect
import os


def load_specific_config(keys, master_config_file='master_config.ini', data_config_file='config.sit.ini'):
    """
    根据调用者的文件名加载特定配置。
    参数:
        keys: list[str] - 需要加载的配置项名称列表，例如 ['MODEL', 'BASE_URL']
        master_config_file: str - 主配置文件路径
        data_config_file: str - 数据配置文件路径
    返回:
        dict - 包含请求的配置项及其值的字典
    """
    # 获取调用者的文件名
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    section = 'CONFIG_' + os.path.splitext(os.path.basename(caller_file))[0].upper() + '_PY'
    
    # 读取 master_config.ini
    master_config = configparser.ConfigParser()
    master_config.read(master_config_file)
    
    if section not in master_config:
        raise ValueError(f"No configuration found for section: {section}")
    
    # 获取该文件对应的服务
    service = master_config[section]['SERVICE']
    
    # 读取 config.sit.ini
    data_config = configparser.ConfigParser()
    data_config.read(data_config_file)
    
    if service not in data_config:
        raise ValueError(f"No data configuration found for service: {service}")
    
    # 根据传入的 keys 从 config.sit.ini 获取配置项
    config_dict = {}
    for key in keys:
        value = data_config[service].get(key, f'default_{key.lower()}')
        # 将 "None" 或 "null" 转换为 None
        config_dict[key] = None if value.lower() in ("none", "null") else value
    
    return config_dict  


def load_other_config():
    # env = os.environ.get('AGENT_ENV')
    config = configparser.ConfigParser()
    # if not env:
    #     env = 'dev'
    # config.read('config.%s.ini'% env)
    config.read('config.sit.ini')
    return config


config = load_other_config()

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
