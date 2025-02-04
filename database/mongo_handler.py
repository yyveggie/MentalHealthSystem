from typing import Dict, Any, Optional
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

class MongoDBStorage:
    def __init__(self, host: str = 'localhost', port: int = 27017):
        """初始化MongoDB连接
        
        Args:
            host: MongoDB服务器地址
            port: MongoDB端口号
        """
        self.client = MongoClient(host, port)
        
    def save_data(self, 
                  database: str,
                  collection: str, 
                  data: pd.DataFrame,
                  is_anonymized: bool = False,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存数据到MongoDB
        
        Args:
            database: 数据库名称
            collection: 集合名称
            data: 要保存的数据（pandas DataFrame）
            is_anonymized: 是否为脱敏数据
            metadata: 额外的元数据信息
        """
        db = self.client[database]
        coll = db[collection]
        
        # 转换DataFrame为字典列表
        records = data.to_dict('records')
        
        # 添加元数据
        timestamp = datetime.now()
        for record in records:
            record['_metadata'] = {
                'timestamp': timestamp,
                'is_anonymized': is_anonymized,
                **(metadata or {})
            }
        
        # 批量插入数据
        try:
            result = coll.insert_many(records)
            print(f"成功保存 {len(result.inserted_ids)} 条记录到 {database}.{collection}")
        except Exception as e:
            print(f"保存数据时发生错误: {str(e)}")
            raise
            
    def close(self):
        """关闭MongoDB连接"""
        self.client.close()