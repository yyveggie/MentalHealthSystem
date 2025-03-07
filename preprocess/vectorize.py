import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from load_config import (
    API_KEY, 
    MONGODB_HOST, 
    MONGODB_PORT,  
    MONGODB_DB_NAME, 
    MONGODB_COLLECTION_NAME,
    MONGODB_FEATURES,
    CASE_HISTORY_BASE_DIRECTOR
    )

os.environ['OPENAI_API_KEY'] = API_KEY

class PatientDataVectorizer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        self.db = self.client[MONGODB_DB_NAME]
        self.collection = self.db[MONGODB_COLLECTION_NAME]
        self.feature_columns = MONGODB_FEATURES

    def vectorize_and_store(self):
        print(f"要处理的特征列表: {self.feature_columns}")
        print(f"数据库中的总文档数: {self.collection.count_documents({})}")
        
        for feature in self.feature_columns:
            doc_count = self.collection.count_documents({feature: {"$exists": True}})
            print(f"特征 '{feature}' 存在的文档数: {doc_count}")
            
            documents = []
            document_ids = []
            
            cursor = self.collection.find({"patient_id": {"$exists": True}})
            
            for doc in cursor:
                # 安全地获取值并转换为字符串
                value = doc.get(feature, "")
                text = str(value) if value is not None else ""
                
                print(f"找到文档: patient_id={doc['patient_id']}, {feature}={text[:50]} (类型: {type(value)})")
                
                doc_id = f"{doc['patient_id']}_{feature}"
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "patient_id": doc["patient_id"],
                        "feature": feature,
                    }
                ))
                document_ids.append(doc_id)
            
            if not documents:
                print(f"Warning: No documents found for feature {feature}")
                continue
                
            persist_directory = os.path.join(CASE_HISTORY_BASE_DIRECTOR, feature)
            os.makedirs(persist_directory, exist_ok=True)

            Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                ids=document_ids
            )

    def close_connection(self):
        self.client.close()

if __name__ == "__main__":
    vectorizer = PatientDataVectorizer()
    vectorizer.vectorize_and_store()
    vectorizer.close_connection()