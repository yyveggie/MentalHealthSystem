import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import json
from datetime import datetime
from typing import Dict, List
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from preprocess.structurer import ExternalInputProcessor

from config_loader import (
    load_specific_config, 
    MONGODB_COLLECTION_NAME, 
    MONGODB_PORT,
    CASE_HISTORY_BASE_DIRECTOR,
    MONGODB_DB_NAME,
    MONGODB_HOST
)

config = load_specific_config(['API_KEY'])
API_KEY = config['API_KEY']

KEYWORDS_FEATURES = ["现病史", "既往史", "过敏史"]   # 用于关键词匹配
VECTOR_FEATURES = ["诊疗经过", "体格检查"]   # 用于向量相似度计算


class TwoStageRetrieval:
    def __init__(self):
        # 原有的初始化代码保持不变
        self.client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        self.db = self.client[MONGODB_DB_NAME]
        self.collection = self.db[MONGODB_COLLECTION_NAME]
        
        # 初始化结构化处理器
        self.structured_processor = ExternalInputProcessor(output_mode="all_features")
        
        # 初始化向量存储，扩展为包含所有特征
        self.embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
        self.vector_stores = {}
        all_features = VECTOR_FEATURES + KEYWORDS_FEATURES
        for feature in all_features:
            persist_directory = os.path.join(CASE_HISTORY_BASE_DIRECTOR, feature)
            if os.path.exists(persist_directory):
                self.vector_stores[feature] = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )

    def _vector_only_retrieval(self, query_texts: Dict[str, str], k: int) -> List[Dict]:
        """
        仅基于向量相似度的检索方案（方案B）
        
        Args:
            query_texts: 包含各个特征文本的字典
            k: 返回结果数量
        
        Returns:
            List[Dict]: 检索结果列表，包含完整的病例文档
        """
        print("\n=== 执行方案B：纯向量相似度检索 ===")
        
        # 存储所有文档的总体相似度分数
        doc_scores = {}
        valid_features_count = {}
        
        # 对每个特征进行向量相似度计算
        all_features = set(VECTOR_FEATURES + KEYWORDS_FEATURES)
        for feature in all_features:
            if feature not in query_texts or feature not in self.vector_stores:
                continue
                
            try:
                print(f"[DEBUG] 开始对特征 {feature} 进行向量相似度检索")
                results = self.vector_stores[feature].similarity_search_with_score(
                    query_texts[feature],
                    k=k * 2  # 获取更多候选以增加找到有效结果的概率
                )
                print(f"[DEBUG] 特征 {feature} 检索完成，返回 {len(results)} 条结果")
                
                # 更新每个文档的累计分数
                for doc, score in results:
                    patient_id = doc.metadata.get('patient_id')
                    if patient_id:
                        similarity = 1 / (1 + score)  # 转换分数为相似度
                        if patient_id not in doc_scores:
                            doc_scores[patient_id] = 0
                            valid_features_count[patient_id] = 0
                        doc_scores[patient_id] += similarity
                        valid_features_count[patient_id] += 1
                        
            except Exception as e:
                print(f"处理特征 {feature} 时出错: {str(e)}")
                continue
        
        # 计算平均相似度并排序
        final_scores = []
        for patient_id, total_score in doc_scores.items():
            if valid_features_count[patient_id] > 0:
                avg_score = total_score / valid_features_count[patient_id]
                final_scores.append({
                    "patient_id": patient_id,
                    "similarity": avg_score
                })
        
        # 按相似度降序排序
        sorted_results = sorted(final_scores, key=lambda x: x["similarity"], reverse=True)
        
        # 获取完整的病例文档
        final_results = []
        for result in sorted_results[:k]:
            doc = self.collection.find_one({"patient_id": result["patient_id"]})
            if doc:
                doc.pop("_id", None)  # 移除 MongoDB 的 _id 字段
                doc["similarity"] = result["similarity"]  # 添加相似度
                doc["rank"] = len(final_results) + 1  # 添加排名
                final_results.append(doc)
        
        return final_results
    
    def _build_entity_query(self, structured_features: Dict[str, Dict]) -> List[str]:
        """构建实体查询条件并返回按匹配数量排序的文档ID列表"""
        print("\n=== 开始构建查询条件 ===")
        
        doc_match_counts = {}
        doc_matched_fields = {}
        
        for feature, structure in structured_features.items():
            print(f"\n处理特征 '{feature}' 的结构化数据:")
            print(json.dumps(structure, ensure_ascii=False, indent=2))
            
            for field, value in structure.items():
                if value is None or (isinstance(value, (list, str)) and not value):
                    continue
                    
                if isinstance(value, list):
                    query = {f"{feature}_结构化.{field}": {"$in": value}}
                    print(f"\n字段 '{field}' 的列表查询条件:")
                    print(json.dumps(query, ensure_ascii=False, indent=2))
                    
                elif isinstance(value, str) and value.strip():
                    query = {f"{feature}_结构化.{field}": {"$regex": value, "$options": "i"}}
                    print(f"\n字段 '{field}' 的字符串查询条件:")
                    print(json.dumps(query, ensure_ascii=False, indent=2))
                else:
                    continue
                    
                matching_docs = self.collection.find(query, {"patient_id": 1})
                matching_docs_list = list(matching_docs)
                print(f"匹配到 {len(matching_docs_list)} 个文档")
                
                for doc in matching_docs_list:
                    doc_id = doc["patient_id"]
                    doc_match_counts[doc_id] = doc_match_counts.get(doc_id, 0) + 1
                    if doc_id not in doc_matched_fields:
                        doc_matched_fields[doc_id] = []
                    doc_matched_fields[doc_id].append(f"{feature}.{field}")
        
        sorted_docs = sorted(doc_match_counts.items(), key=lambda x: x[1], reverse=True)
        
        print("\n=== 匹配结果详情 ===")
        for doc_id, match_count in sorted_docs:
            print(f"\n文档 {doc_id}:")
            print(f"- 匹配字段数: {match_count}")
            print(f"- 匹配的字段: {', '.join(doc_matched_fields[doc_id])}")
        
        return [doc_id for doc_id, _ in sorted_docs]
        
    def retrieve_similar_cases(self, query_texts: Dict[str, str], scheme: str = 'A', n: int = 10, k: int = 5) -> List[Dict]:
        """
        检索相似病例，支持选择检索方案
        
        Args:
            query_texts: 输入的查询文本
            scheme: 检索方案，'A' 为两阶段检索，'B' 为纯向量相似度检索
            n: 选择实体匹配排名前n的文档进行向量相似度计算（仅方案A）
            k: 最终返回相似度最高的前k个文档
        
        Returns:
            List[Dict]: 检索到的完整病例文档列表
        """
        try:
            if scheme == 'A':
                print("\n=== 使用方案A：两阶段检索 ===")
                results = self._original_two_stage_retrieval(query_texts, n, k)
            elif scheme == 'B':
                print("\n=== 使用方案B：纯向量相似度检索 ===")
                results = self._vector_only_retrieval(query_texts, k)
            else:
                return {"error": "无效的方案参数"}
            
            if not results:
                return {"error": "未能找到任何有效结果"}
            
            return results
        except Exception as e:
            return {"error": f"检索过程中出错: {str(e)}"}

    def _original_two_stage_retrieval(self, query_texts: Dict[str, str], n: int, k: int) -> List[Dict]:
        """
        原有的两阶段检索方案（方案A）
        
        Args:
            query_texts: 输入的查询文本
            n: 选择实体匹配排名前n的文档进行向量相似度计算
            k: 最终返回相似度最高的前k个文档
        
        Returns:
            List[Dict]: 检索结果列表，包含完整的病例文档
        """
        try:
            # 第一阶段：实体特征匹配
            structured_features = {}
            for feature in KEYWORDS_FEATURES:
                if feature in query_texts:
                    try:
                        structured = self.structured_processor.process_single_text(
                            query_texts[feature], 
                            feature
                        )
                        structured_features[feature] = structured
                    except Exception as e:
                        print(f"结构化处理 {feature} 时出错: {str(e)}")
                        continue
                
            if not structured_features:
                return []

            candidate_ids = self._build_entity_query(structured_features)
            
            if not candidate_ids:
                return []
                
            print(f"\n实体匹配找到的文档数: {len(candidate_ids)}")
            print(f"选择匹配度最高的前 {n} 个文档进行向量相似度计算")
            
            # 第二阶段：向量相似度计算
            vector_scores = []
            for doc_id in candidate_ids[:n]:
                total_score = 0
                valid_features = 0
                
                for feature in VECTOR_FEATURES:
                    if feature not in query_texts or feature not in self.vector_stores:
                        continue
                    
                    results = self.vector_stores[feature].similarity_search_with_score(
                        query_texts[feature],
                        k=1,
                        filter={"patient_id": doc_id}
                    )
                    
                    if results:
                        score = 1 / (1 + results[0][1])
                        total_score += score
                        valid_features += 1
                
                if valid_features > 0:
                    avg_score = total_score / valid_features
                    vector_scores.append({
                        "patient_id": doc_id,
                        "similarity": avg_score
                    })
            
            sorted_cases = sorted(vector_scores, key=lambda x: x["similarity"], reverse=True)
            
            # 获取完整的病例文档
            final_results = []
            for case in sorted_cases[:k]:
                doc = self.collection.find_one({"patient_id": case["patient_id"]})
                if doc:
                    doc.pop("_id", None)
                    doc["similarity"] = case["similarity"]
                    doc["rank"] = len(final_results) + 1
                    final_results.append(doc)
            
            print(f"\n方案A找到 {len(final_results)} 个有效结果")
            return final_results
            
        except Exception as e:
            print(f"两阶段检索过程中出错: {str(e)}")
            return []
    
    def close(self):
        """安全关闭连接"""
        self.client.close()
        if hasattr(self.structured_processor, "close"):
            self.structured_processor.close()


if __name__ == "__main__":
    retrieval_system = TwoStageRetrieval()
    try:
        test_query = {
            "现病史": "患者于2020.02起出现胸闷不适，渐出现双手及后背发冷，伴有双手发麻，双下肢无力，反复担心自己得了不治之症，由此出现坐立不安、心神不宁，听到大的声响后即出现心慌不适，伴有入睡困难，多次至外院就诊，完善检查未见明显异常，患者对此半信半疑，后胸闷、发冷等症状持续不能缓解，因此感到心情烦躁，疲乏无力，伴有纳差明显，1月内体重下降4kg，更加担心自己的健康情况，于半月前至我科门诊就诊，考虑'焦虑状态'，予以舍曲林、阿普唑仑等药物治疗，患者规律服药，诉心情烦躁及胸闷、发冷等症状较前有好转。患者于2天前因症状好转自行停用阿普唑仑，再次出现上述症状加重。现为进一步诊治，门诊以'焦虑状态'收入我科。    病程中，胃纳差，夜眠差，二便基本正常，体重1月内减轻4kg。否认消极，无冲动，伤人，毁物，外跑行为。",
            "既往史": "高血压病史10年，规律服用降压药物",
            "诊疗经过": "给予利尿、强心等治疗",
            "体格检查": "双肺呼吸音粗，可闻及湿性啰音"
        }
        
        results = retrieval_system.retrieve_similar_cases(
            query_texts=test_query,
            scheme='B',  # 可切换为 'A' 测试两阶段检索
            k=3
        )
        
        # 定义一个自定义序列化函数，处理 datetime 对象
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()  # 将 datetime 转换为 ISO 格式字符串
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # 在 json.dumps 中使用 default 参数
        results_json = json.dumps(results, ensure_ascii=False, indent=2, default=json_serial)
        print("Retrieved enhanced results:")
        print(results_json)
        
        for i in json.loads(results_json):
            print(i["出院诊断"])
        
    finally:
        retrieval_system.close()