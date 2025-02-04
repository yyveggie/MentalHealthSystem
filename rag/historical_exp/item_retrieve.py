import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from typing import Dict, List, Union
from pymongo import MongoClient
import json
from datetime import datetime
from bson import ObjectId
from enum import Enum


class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # 对于其他不可序列化的类型，转换为字符串

class QueryOperator(str, Enum):
    AND = "AND"
    OR = "OR"

class FeatureRetrieval:
    def __init__(self, host: str, port: int, db_name: str, collection_name: str):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.structure_file = "./database/feature_structures.json"
        self._load_structure()
    
    def _load_structure(self):
        """加载特征结构"""
        if not os.path.exists(self.structure_file):
            raise FileNotFoundError(f"特征结构文件 {self.structure_file} 不存在，请先运行 FeatureStructureManager")
        
        with open(self.structure_file, 'r', encoding='utf-8') as f:
            self.feature_structures = json.load(f)
    
    def validate_query(self, feature_queries: List[Dict[str, Dict[str, Union[str, List[str]]]]]) -> bool:
        """验证查询条件是否符合特征结构"""
        for query in feature_queries:
            for feature, conditions in query.items():
                # 只检查特征是否存在，不再检查结构化字段
                if feature not in self.feature_structures and f"{feature}_结构化" not in self.feature_structures:
                    print(f"警告: 特征 '{feature}' 和 '{feature}_结构化' 都不存在")
                    return False
        return True

    def search_by_feature(self, 
                        feature_queries: List[Dict[str, Dict[str, Union[str, List[str]]]]],
                        operator: QueryOperator = QueryOperator.AND) -> List[Dict]:
        """搜索符合条件的文档"""
        # 验证查询条件
        if not self.validate_query(feature_queries):
            return []
        
        # 构建查询条件
        query_conditions = []
        for feature_query in feature_queries:
            for feature, conditions in feature_query.items():
                feature_conditions = []
                
                # 处理嵌套字典的递归函数
                def build_conditions(prefix, value):
                    if isinstance(value, dict):
                        dict_conditions = []
                        for k, v in value.items():
                            dict_conditions.extend(build_conditions(f"{prefix}.{k}", v))
                        return dict_conditions
                    elif isinstance(value, list):
                        if value and isinstance(value[0], dict):
                            # 处理字典列表
                            list_conditions = []
                            for dict_item in value:
                                dict_conditions = []
                                for k, v in dict_item.items():
                                    dict_conditions.extend(build_conditions(f"{prefix}.{k}", v))
                                if dict_conditions:
                                    list_conditions.append({"$or": dict_conditions})
                            return list_conditions
                        else:
                            return [{prefix: {"$in": value}}]
                    else:
                        return [{prefix: value}]
                
                # 构建查询条件
                for field, value in conditions.items():
                    feature_path = feature  # 使用原始特征名
                    field_conditions = build_conditions(f"{feature_path}.{field}", value)
                    feature_conditions.extend(field_conditions)
                
                if feature_conditions:
                    query_conditions.append({
                        "$or": feature_conditions
                    })
        
        # 组合查询条件
        if operator == QueryOperator.AND:
            final_query = {"$and": query_conditions} if query_conditions else {}
        else:
            final_query = {"$or": query_conditions} if query_conditions else {}
        
        # 执行查询
        print("\n=== 执行查询 ===")
        print("查询条件:", json.dumps(final_query, ensure_ascii=False, indent=2))
        
        results = list(self.collection.find(final_query))
        print(f"找到 {len(results)} 个匹配文档")
        
        return results
    
    def close(self):
        if hasattr(self, 'client'):
            self.client.close()
            
def print_results(results):
            
    try:
        for i in range(len(results)):
            print(f"\n第 {i+1} 个匹配文档:")
            print(json.dumps(
                results[i], 
                ensure_ascii=False, 
                indent=2, 
                cls=MongoJSONEncoder
            ))
    except Exception as e:
        print(f"打印文档时出错: {str(e)}")
        # 尝试打印简化版本的文档
        for i in range(len(results)):
            print(f"\n第 {i+1} 个匹配文档的关键字段:")
            doc = results[i]
            # 只打印一些基本字段
            simple_doc = {
                "patient_id": doc.get("patient_id"),
                "现病史": doc.get("现病史"),
                "既往史": doc.get("既往史"),
                "诊断": doc.get("诊断")
            }
            print(json.dumps(simple_doc, ensure_ascii=False, indent=2, cls=MongoJSONEncoder))

def main():
    from load_config import MONGODB_HOST, MONGODB_PORT, MONGODB_DB_NAME, MONGODB_COLLECTION_NAME

    # 使用保存的结构进行查询
    retrieval = FeatureRetrieval(
        host=MONGODB_HOST,
        port=MONGODB_PORT,
        db_name=MONGODB_DB_NAME,
        collection_name=MONGODB_COLLECTION_NAME
    )
    
    try:
        # 示例查询
        # 1. 直接查询结构化字段
        query1 = [
            {
                "现病史_结构化": {
                    "发病方式": ["慢性"]
                }
            }
        ]
        
        results1 = retrieval.search_by_feature(query1, operator=QueryOperator.AND)
        print_results(results1)

        # # 2. 查询包含字典列表的字段
        # query2 = [
        #     {
        #         "既往史_结构化": {
        #             "手术史": [{
        #                 "手术名称": "腰后路减压植骨内固定术"
        #             }]
        #         }
        #     }
        # ]
        
        # results2 = retrieval.search_by_feature(query2, operator=QueryOperator.AND)
        # print("\n交集查询示例1 - 同一特征内多条件：")
        # print(f"找到 {len(results2)} 个匹配文档")

        # # 3. 多条件并集查询
        # query3 = [
        #     {
        #         "现病史_结构化": {
        #             "发病方式": ["突发反复性"]
        #         }
        #     },
        #     {
        #         "既往史_结构化": {
        #             "手术史": [{
        #                 "手术名称": "腰后路减压植骨内固定术"
        #             }]
        #         }
        #     }
        # ]
        
        # results3 = retrieval.search_by_feature(query3, operator=QueryOperator.AND)
        # print("\n交集查询示例1 - 同一特征内多条件：")
        # print(f"找到 {len(results3)} 个匹配文档")
        
        
        # # 示例1：同一个特征内多个条件的交集查询
        # # 查找既有"突发反复性"发病方式，又有"持续性"病程的病例
        # query1 = [
        #     {
        #         "现病史_结构化": {
        #             "发病方式": ["突发反复性"],
        #             "病程": ["持续性"]
        #         }
        #     }
        # ]
        # results1 = retrieval.search_by_feature(query1, operator=QueryOperator.AND)
        # print("\n交集查询示例1 - 同一特征内多条件：")
        # print(f"找到 {len(results1)} 个匹配文档")

        # # 示例2：不同特征间的交集查询
        # # 查找发病方式是"突发反复性"且有特定手术史的病例
        # query2 = [
        #     {
        #         "现病史_结构化": {
        #             "发病方式": ["突发反复性"]
        #         }
        #     },
        #     {
        #         "既往史_结构化": {
        #             "手术史": [{
        #                 "手术名称": "腰后路减压植骨内固定术"
        #             }]
        #         }
        #     }
        # ]
        # results2 = retrieval.search_by_feature(query2, operator=QueryOperator.AND)
        # print("\n交集查询示例2 - 不同特征间交集：")
        # print(f"找到 {len(results2)} 个匹配文档")

        # # 示例3：复杂交集查询
        # # 查找同时满足多个特征和子特征的病例
        # query3 = [
        #     {
        #         "现病史_结构化": {
        #             "发病方式": ["突发反复性"],
        #             "病程": ["持续性"]
        #         }
        #     },
        #     {
        #         "既往史_结构化": {
        #             "手术史": [{
        #                 "手术名称": "腰后路减压植骨内固定术"
        #             }]
        #         }
        #     },
        #     {
        #         "过敏史_结构化": {
        #             "过敏物质": ["青霉素"]
        #         }
        #     }
        # ]
        # results3 = retrieval.search_by_feature(query3, operator=QueryOperator.AND)
        # print("\n交集查询示例3 - 复杂交集：")
        # print(f"找到 {len(results3)} 个匹配文档")

        # # 打印结果
        # for i, doc in enumerate(results3, 1):
        #     print(f"\n文档 {i}:")
        #     print(json.dumps(doc, ensure_ascii=False, indent=2))

    finally:
        retrieval.close()


if __name__ == "__main__":
    main()