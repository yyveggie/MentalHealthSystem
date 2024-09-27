import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from load_config import (
    MONGODB_HOST, 
    MONGODB_PORT, 
    CASE_HISTORY_BASE_DIRECTOR, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTION_NAME,
    MONGODB_FEATURES,
    HUGGINGFACE_EMBEDDING_MODEL
)

class PatientDataVectorizer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
        self.client = MongoClient(MONGODB_HOST, MONGODB_PORT)
        self.db = self.client[MONGODB_DB_NAME]
        self.collection = self.db[MONGODB_COLLECTION_NAME]
        self.feature_columns = MONGODB_FEATURES

    def vectorize_and_store(self):
        for feature in self.feature_columns:
            documents = []
            cursor = self.collection.find({feature: {'$exists': True}})
            for doc in cursor:
                text = doc[feature]
                metadata = {
                    "patient_id": doc["patient_id"],
                    "feature": feature,
                }
                documents.append(Document(page_content=text, metadata=metadata))
            
            persist_directory = os.path.join(CASE_HISTORY_BASE_DIRECTOR, feature)
            os.makedirs(persist_directory, exist_ok=True)

            Chroma.from_documents(documents, embedding=self.embeddings, persist_directory=persist_directory)
            print(f"Feature '{feature}' vectorized and stored.")

    def close_connection(self):
        self.client.close()

if __name__ == "__main__":
    vectorizer = PatientDataVectorizer()
    vectorizer.vectorize_and_store()
    vectorizer.close_connection()