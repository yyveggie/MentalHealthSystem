import logging
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.exceptions import AuthorizationException
import datetime
import os

# 配置Elasticsearch连接信息
#es_host = 'https://node.itingluo.com'  # 替换为你的Elasticsearch地址
username = 'elastic'  # 替换为你的用户名
password = 'Jx4&7uS#9@Lp2Wx'  # 替换为你的密码

# 创建Elasticsearch客户端
http_auth = (username, password)
#conn = RequestsHttpConnection(es_host)

class ElasticsearchHandler(logging.Handler):
    def __init__(self, es_instance, index_name):
        super().__init__()
        self.es = es_instance
        self.index_name = index_name

    def emit(self, record):
        try:
            doc = {
                'timestamp': datetime.datetime.utcnow(),
                'level': record.levelname,
                'message': self.format(record),
                'session_id': getattr(record, 'session_id', None),
                'user_id': getattr(record, 'user_id', None),
                'module': record.module
            }
            self.es.index(index=self.index_name, body=doc)
        except Exception:
            self.handleError(record)

def is_elasticsearch_available():
    try:
        es = Elasticsearch(
            connection_class=RequestsHttpConnection,
            http_auth=(username, password),
            # 替换下面的URL为您的Elasticsearch实例的URL
            hosts=["https://node.itingluo.com"]
        )
        return es.ping()
    except:
        return False

def setup_logging():
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if is_elasticsearch_available():
        # Elasticsearch 可用，使用 Elasticsearch 处理器
        es = Elasticsearch(
            connection_class=RequestsHttpConnection,
            http_auth=(username, password),
            # 替换下面的URL为您的Elasticsearch实例的URL
            hosts=["https://node.itingluo.com"]
            verify_certs=True,  # 如果使用自签名证书，可能需要设置为False
            timeout=10
        )
        es_handler = ElasticsearchHandler(es, 'chat_logs')
        root_logger.addHandler(es_handler)
        print("Using Elasticsearch for logging")
    else:
        # Elasticsearch 不可用，使用文件处理器
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print("Using file logging")

    # 添加控制台处理器（可选）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return None, None  # 不再返回 es 和 listener

# 如果你想完全禁用日志记录，可以使用这个函数替代上面的 setup_logging
def disable_logging():
    logging.disable(logging.CRITICAL)
    print("Logging is disabled")
    return None, None
