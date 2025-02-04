import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from typing import Dict, Any, Optional
from pymongo import MongoClient
import json
from load_config import MONGODB_HOST, MONGODB_PORT, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME
from pydantic import BaseModel, Field


class FeatureStructureManager:
    def __init__(self, host: str, port: int, db_name: str, collection_name: str):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.structure_file = "./database/feature_structures.json"
    
    def extract_and_save_structures(self):
        """提取所有特征结构并保存到文件"""
        feature_structures = {}
        all_docs = self.collection.find({})
        
        print("开始提取特征结构...")
        for doc in all_docs:
            for field, value in doc.items():
                if field in ['_id', 'patient_id']:  # 跳过特定字段
                    continue
                
                if field not in feature_structures:
                    feature_structures[field] = {
                        'raw_structure': self._analyze_structure(value),
                        'structured_fields': {}
                    }
                    
                # 检查是否有对应的结构化字段
                structured_field = f"{field}_结构化"
                if structured_field in doc and isinstance(doc[structured_field], dict):
                    feature_structures[field]['structured_fields'] = self._analyze_structure(doc[structured_field])
        
        # 保存到文件
        with open(self.structure_file, 'w', encoding='utf-8') as f:
            json.dump(feature_structures, f, ensure_ascii=False, indent=2)
        
        print(f"特征结构已保存到 {self.structure_file}")
        return feature_structures
    
    def _analyze_structure(self, value: Any) -> Dict:
        """分析值的结构"""
        structure = {'type': type(value).__name__, 'values': set()}
        
        if isinstance(value, dict):
            substructure = {}
            for k, v in value.items():
                substructure[k] = self._analyze_structure(v)
            return {'type': 'dict', 'fields': substructure}
        
        elif isinstance(value, list):
            # 如果是列表，分析其元素
            if value:
                if all(isinstance(x, dict) for x in value):
                    # 如果列表中都是字典，分析第一个字典的结构
                    return {'type': 'list[dict]', 'element_structure': self._analyze_structure(value[0])}
                else:
                    # 如果是简单类型的列表，记录所有唯一值
                    return {'type': 'list', 'element_type': type(value[0]).__name__}
            return {'type': 'list', 'element_type': 'unknown'}
        
        else:
            # 简单类型
            return {'type': type(value).__name__}
    
    def print_structure_summary(self):
        """打印特征结构摘要"""
        if not os.path.exists(self.structure_file):
            print("结构文件不存在，请先运行 extract_and_save_structures()")
            return
        
        with open(self.structure_file, 'r', encoding='utf-8') as f:
            structures = json.load(f)
        
        print("\n=== 特征结构摘要 ===")
        for feature, info in structures.items():
            print(f"\n特征: {feature}")
            print("  原始数据结构:")
            self._print_structure(info['raw_structure'], indent=4)
            
            if info['structured_fields']:
                print("  结构化字段:")
                self._print_structure(info['structured_fields'], indent=4)
    
    def _print_structure(self, structure: Dict, indent: int = 0):
        """递归打印结构"""
        prefix = " " * indent
        if structure['type'] == 'dict':
            for field, field_structure in structure['fields'].items():
                print(f"{prefix}{field}:")
                self._print_structure(field_structure, indent + 2)
        else:
            print(f"{prefix}类型: {structure['type']}")
            if 'element_type' in structure:
                print(f"{prefix}元素类型: {structure['element_type']}")
    
    def close(self):
        if hasattr(self, 'client'):
            self.client.close()

class MedicalHistory(BaseModel):
    # 假设之前是:
    # 传染病史: Dict[str, Any]
    # 导致在传入 None 时报错
    # 现修改为可选字段 + 默认值:
    传染病史: Optional[Dict[str, Any]] = None

    # 如果你有其他字段也可能为空, 也可 similarly 改为可选
    # ...

def main():
    structure_manager = FeatureStructureManager(
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        db_name=MONGODB_DB_NAME,
        collection_name=MONGODB_COLLECTION_NAME
    )
    
    try:
        structure_manager.extract_and_save_structures()
        structure_manager.print_structure_summary()
    finally:
        structure_manager.close()

if __name__ == "__main__":
    main()

