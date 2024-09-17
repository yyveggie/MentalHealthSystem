import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import json
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import logging
from logging_config import setup_logging

from load_config import (
    OPENAI_API_KEY,
    MONGODB_HOST, 
    MONGODB_PORT, 
    MONGODB_BASE_DIRECTOR, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTION_NAME,
    MONGODB_FEATURES
)

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

logger = logging.getLogger(__name__)
es = setup_logging()

class PatientDiagnosisAPI:
    def __init__(self):
        self.client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        self.db = self.client[MONGODB_DB_NAME]
        self.collection = self.db[MONGODB_COLLECTION_NAME]
        self.feature_columns = MONGODB_FEATURES
        self.embeddings = OpenAIEmbeddings()
        self.vectorstores = {}
        self.load_vectorstores()
        logger.info("PatientDiagnosisAPI initialized")

    def load_vectorstores(self) -> None:
        for feature in self.feature_columns:
            persist_directory = os.path.join(MONGODB_BASE_DIRECTOR, feature)
            if os.path.exists(persist_directory):
                self.vectorstores[feature] = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
                logger.info(f"Loaded vectorstore for feature '{feature}'")
            else:
                logger.warning(f"Vectorstore for feature '{feature}' not found. Please run vectorization first.")

    def query_by_feature(self, feature: str, query_text: str, k: int = 3) -> list:
        if feature not in self.vectorstores:
            logger.warning(f"Vectorstore for feature '{feature}' not available")
            return []
        
        logger.info(f"Querying feature '{feature}' with text: {query_text[:50]}...")  # Log only first 50 chars for brevity
        results = self.vectorstores[feature].similarity_search_with_score(query_text, k=k)
        
        diagnoses = []
        for result, score in results:
            patient_id = result.metadata["patient_id"]
            doc = self.collection.find_one({"patient_id": patient_id}, {"出院诊断": 1})
            discharge_diagnosis = doc.get("出院诊断", "该病例尚未提供诊断信息")
            diagnoses.append({"diagnosis": discharge_diagnosis, "similarity": float(score)})
        
        logger.info(f"Found {len(diagnoses)} similar diagnoses for feature '{feature}'")
        return diagnoses

    def process_query(self, query_json: str) -> str:
        query_dict = json.loads(query_json)
        result = {}
        
        logger.info(f"Processing query with {len(query_dict)} features")
        
        for feature, query_text in query_dict.items():
            if feature in self.feature_columns:
                diagnoses = self.query_by_feature(feature, query_text)
                result[feature] = diagnoses
            else:
                logger.warning(f"Unsupported feature '{feature}' in query")
        
        logger.info("Query processing completed")
        return json.dumps(result, ensure_ascii=False, indent=2)

    def close_connection(self) -> None:
        self.client.close()
        logger.info("MongoDB connection closed")

if __name__ == "__main__":
    api = PatientDiagnosisAPI()
    
    query_json = json.dumps({
        "过敏史": "药物过敏史：未发现；食物过敏史：否认",
        "诊疗经过":"患者2023年09月21日08时56分入院，入院后完善相关检查，予以文拉法辛缓释胶囊75mg qd、氯硝西泮0.5mg bid、阿立哌唑5mg qn、右佐匹克隆3mg qm改善情绪及睡眠等治疗，患者病情稳定，过程顺利，于2023年09月30日10时18分出院。"
    })
    
    logger.info("Starting query processing")
    result = api.process_query(query_json)
    logger.info("Query processing finished")
    # print(result)
    
    api.close_connection()