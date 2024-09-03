import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import json
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

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

class PatientDiagnosisAPI:
    def __init__(self):
        self.client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        self.db = self.client[MONGODB_DB_NAME]
        self.collection = self.db[MONGODB_COLLECTION_NAME]
        self.feature_columns = MONGODB_FEATURES
        self.embeddings = OpenAIEmbeddings()
        self.vectorstores = {}
        self.load_vectorstores()

    def load_vectorstores(self) -> None:
        """
        Load vector stores for each feature from the specified directory.
        """
        for feature in self.feature_columns:
            persist_directory = os.path.join(MONGODB_BASE_DIRECTOR, feature)
            if os.path.exists(persist_directory):
                self.vectorstores[feature] = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
            else:
                print(f"Vectorstore for feature '{feature}' not found. Please run vectorization first.")

    def query_by_feature(self, feature: str, query_text: str, k: int = 3) -> list:
        """
        Query the vector store for a specific feature and return the top k similar diagnoses.

        Args:
            feature (str): The feature to query.
            query_text (str): The query text.
            k (int, optional): The number of results to return. Defaults to 3.

        Returns:
            list: A list of dictionaries containing diagnoses and their similarity scores.
        """
        if feature not in self.vectorstores:
            return []
        
        results = self.vectorstores[feature].similarity_search_with_score(query_text, k=k)
        
        diagnoses = []
        for result, score in results:
            patient_id = result.metadata["patient_id"]
            doc = self.collection.find_one({"patient_id": patient_id}, {"出院诊断": 1})
            discharge_diagnosis = doc.get("出院诊断", "该病例尚未提供诊断信息")
            diagnoses.append({"diagnosis": discharge_diagnosis, "similarity": float(score)})
        
        return diagnoses

    def process_query(self, query_json: str) -> str:
        """
        Process a JSON query string and return the results as a JSON string.

        Args:
            query_json (str): A JSON string containing the query.

        Returns:
            str: A JSON string containing the query results.
        """
        query_dict = json.loads(query_json)
        result = {}
        
        for feature, query_text in query_dict.items():
            if feature in self.feature_columns:
                diagnoses = self.query_by_feature(feature, query_text)
                result[feature] = diagnoses
        
        return json.dumps(result, ensure_ascii=False, indent=2)

    def close_connection(self) -> None:
        """
        Close the MongoDB connection.
        """
        self.client.close()

if __name__ == "__main__":
    api = PatientDiagnosisAPI()
    
    query_json = json.dumps({
        "过敏史": "药物过敏史：未发现；食物过敏史：否认",
        "诊疗经过":"患者2023年09月21日08时56分入院，入院后完善相关检查，予以文拉法辛缓释胶囊75mg qd、氯硝西泮0.5mg bid、阿立哌唑5mg qn、右佐匹克隆3mg qm改善情绪及睡眠等治疗，患者病情稳定，过程顺利，于2023年09月30日10时18分出院。"
    })
    
    result = api.process_query(query_json)
    print(result)
    
    api.close_connection()