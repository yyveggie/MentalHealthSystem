from pymongo import MongoClient
from typing import Dict, List, Any

class MongoDBPatientInfoSystem:
    def __init__(self, connection_string: str):
        self.client = MongoClient(connection_string)

    def get_user_db(self, user_id: str):
        return self.client[user_id]

    def get_category_collection(self, user_id: str, category: str):
        db = self.get_user_db(user_id)
        return db[category]

    def add_memory(self, user_id: str, category: str, memory: Dict[str, Any]):
        collection = self.get_category_collection(user_id, category)
        result = collection.insert_one(memory)
        return result.inserted_id

    def update_memory(self, user_id: str, category: str, memory_id: str, updated_memory: Dict[str, Any]):
        collection = self.get_category_collection(user_id, category)
        result = collection.update_one({"_id": memory_id}, {"$set": updated_memory})
        return result.modified_count

    def delete_memory(self, user_id: str, category: str, memory_id: str):
        collection = self.get_category_collection(user_id, category)
        result = collection.delete_one({"_id": memory_id})
        return result.deleted_count

    def get_memories(self, user_id: str, category: str) -> List[Dict[str, Any]]:
        collection = self.get_category_collection(user_id, category)
        return list(collection.find())

# 可以添加更多的方法，比如批量操作、聚合查询等，根据需求扩展

if __name__ == "__main__":
    # 这里可以添加一些测试代码
    db_system = MongoDBPatientInfoSystem("mongodb://localhost:27017/")
    
    # 测试添加记忆
    memory_id = db_system.add_memory("test_user", "CHIEF_COMPLAINT", {"content": "Patient reported severe headache"})
    
    # 测试获取记忆
    memories = db_system.get_memories("test_user", "CHIEF_COMPLAINT")
    print("Retrieved memories:", memories)
    
    # 测试更新记忆
    db_system.update_memory("test_user", "CHIEF_COMPLAINT", memory_id, {"content": "Patient reported severe headache and nausea"})
    
    # 测试删除记忆
    db_system.delete_memory("test_user", "CHIEF_COMPLAINT", memory_id)