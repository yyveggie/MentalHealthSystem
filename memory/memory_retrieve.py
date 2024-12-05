import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Dict, List, Optional
from pymongo import MongoClient
from typing import Dict, List, TypedDict, Union
from load_config import MONGODB_HOST, MONGODB_PORT


class MemoryInfo(TypedDict):
    knowledge: str
    knowledge_old: Union[str, None]
    timestamp: str

class MemoryRetrievalSystem:
    def __init__(self):
        """初始化记忆检索系统，自动连接到MongoDB"""
        self.client = MongoClient(
            host=MONGODB_HOST,
            port=MONGODB_PORT,
            directConnection=True
        )

    def retrieve_memories_by_categories(
        self,
        user_id: str,
        categories: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, List[dict]]:
        """
        根据类别检索最新记忆
        
        Args:
            user_id: 用户ID
            categories: 要检索的类别列表，如果为None则检索所有类别
            confidence_threshold: 置信度阈值，如果指定则只返回高于此阈值的记忆
            
        Returns:
            Dict[str, List[dict]]: 按类别组织的最新记忆字典
        """
        result = {}
        db = self.client[user_id]
        
        # 如果未指定类别，则获取数据库中所有可用的类别
        if categories is None:
            categories = db.list_collection_names()

        # 从每个类别集合中获取文档
        for category in categories:
            collection = db[category]
            
            # 构建查询条件
            query = {"is_latest": True}  # 只获取最新版本
            if confidence_threshold is not None:
                query["confidence"] = {"$gte": confidence_threshold}
                
            try:
                # 执行查询
                latest_memories = list(collection.find(query))
                
                if latest_memories:
                    # 处理查询结果
                    memories = []
                    for doc in latest_memories:
                        doc_copy = doc.copy()
                        if "_id" in doc_copy:
                            doc_copy["_id"] = str(doc_copy["_id"])
                        memories.append(doc_copy)
                    
                    result[category] = memories
                    
            except Exception as e:
                print(f"Error processing category '{category}': {str(e)}")
                continue
                
        return result

    def retrieve_memory_history(
        self,
        user_id: str,
        category: str,
        memory_id: str
    ) -> List[dict]:
        """
        获取特定记忆的历史版本
        
        Args:
            user_id: 用户ID
            category: 记忆类别
            memory_id: 记忆ID
            
        Returns:
            List[dict]: 记忆的历史版本列表，按时间倒序排列
        """
        collection = self.client[user_id][category]
        history = []
        current_id = memory_id
        
        while current_id:
            try:
                memory = collection.find_one({"_id": current_id})
                if memory:
                    memory_copy = memory.copy()
                    memory_copy["_id"] = str(memory_copy["_id"])
                    history.append(memory_copy)
                    current_id = memory.get('previous_version_id')
                else:
                    break
            except Exception as e:
                print(f"Error retrieving memory history: {str(e)}")
                break
                
        return history

    def parse_memory_result(self, memory_result: Dict[str, List[dict]]) -> Dict[str, List[MemoryInfo]]:
        """
        解析记忆检索结果，提取每条记忆的knowledge、knowledge_old和timestamp信息
        
        Args:
            memory_result: 原始记忆检索结果
            
        Returns:
            Dict[str, List[MemoryInfo]]: 解析后的记忆信息，按类别组织
        """
        parsed_result = {}
        
        for category, memories in memory_result.items():
            parsed_memories = []
            for memory in memories:
                memory_info = {
                    "knowledge": memory["knowledge"],
                    "knowledge_old": memory.get("knowledge_old"),  # 使用get处理可能不存在的字段
                    "timestamp": memory["timestamp"]
                }
                parsed_memories.append(memory_info)
            
            parsed_result[category] = parsed_memories
            
        return parsed_result


if __name__ == "__main__":
    # 测试代码
    memory_system = MemoryRetrievalSystem()
    user_id = "test"

    # 检索特定类别的记忆
    memories = memory_system.retrieve_memories_by_categories(
        user_id=user_id,
        categories=["情绪体验", "行为模式", "主诉", "人口学信息"],
        confidence_threshold=0.8
    )
    print(f"用户{user_id}的记忆:", memories)
    print("处理后形式:", memory_system.parse_memory_result(memories))

    # 获取记忆历史
    memory_history = memory_system.retrieve_memory_history(
        user_id="test",
        category="主诉",
        memory_id="some_memory_id"
    )