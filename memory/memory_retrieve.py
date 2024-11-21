import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from typing import Dict, List, Optional
from pymongo import MongoClient
from load_config import MONGODB_HOST, MONGODB_PORT


class MemoryRetrievalSystem:
    def __init__(self):
        print(f"正在连接到MongoDB: mongodb://{MONGODB_HOST}:{MONGODB_PORT}/")
        self.client = MongoClient(
            host=MONGODB_HOST,
            port=MONGODB_PORT,
            directConnection=True
        )
        print("MongoDB连接已建立")
        
    def retrieve_memories_by_categories(
        self,
        categories: List[str],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, List[dict]]:
        """
        根据类别列表检索记忆
        """
        result = {}
        db = self.client.test
        
        print(f"\n数据库中的所有集合: {db.list_collection_names()}")
        print(f"要检索的类别: {categories}")
        
        # 从每个类别集合中获取所有文档
        for category in categories:
            print(f"\n处理类别 '{category}':")
            collection = db[category]
            
            # 构建查询条件(只检查置信度)
            query = {}
            if confidence_threshold is not None:
                query["confidence"] = {"$gte": confidence_threshold}
                
            try:
                # 获取所有文档
                all_docs = list(collection.find(query))
                print(f"在集合 '{category}' 中找到 {len(all_docs)} 条记录")
                
                if all_docs:
                    # 转换ObjectId为字符串
                    memories = []
                    for doc in all_docs:
                        if "_id" in doc:
                            doc["_id"] = str(doc["_id"])
                        memories.append(doc)
                    
                    result[category] = memories
                    print(f"成功添加 {len(memories)} 条记忆到结果中")
                    
            except Exception as e:
                print(f"处理类别 '{category}' 时出错: {str(e)}")
                continue
                
        return result

    def print_memories_json(
        self,
        memories_dict: Dict[str, List[dict]],
        indent: int = 2
    ) -> None:
        """
        以JSON格式打印记忆
        """
        if not memories_dict:
            print("\n未找到任何记忆")
            return
            
        print("\n检索到的记忆:")
        print(json.dumps(memories_dict, ensure_ascii=False, indent=indent, default=str))

def main():
    retrieval_system = MemoryRetrievalSystem()
    
    # 检索特定类别的记忆
    categories = ["情绪状态", "人际动力"]
    
    memories = retrieval_system.retrieve_memories_by_categories(categories)
    retrieval_system.print_memories_json(memories)

if __name__ == "__main__":
    main()