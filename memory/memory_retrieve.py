import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from memory import explicit_memory, implicit_memory
from typing import List, Dict, Any

import asyncio

import warnings
warnings.filterwarnings("ignore")

class ConcurrentMemorySearch:
    def __init__(self, explicit_memory_query, implicit_memory_query, user_id) -> None:
        self.explicit_memory_query = explicit_memory_query
        self.implicit_memory_query = implicit_memory_query
        self.user_id = user_id

    def extract_memory_content(self, memories: List[Dict[str, Any]]) -> List[str]:
        return [memory.get('memory', '') for memory in memories if 'memory' in memory]

    async def search_explicit_memory(self) -> List[str]:
        if self.explicit_memory_query:
            e_memories = await explicit_memory.search_patient_info(self.user_id, self.explicit_memory_query)
            return self.extract_memory_content(e_memories)
        return []

    async def search_implicit_memory(self) -> List[str]:
        if self.implicit_memory_query:
            i_memories = await implicit_memory.search_mental_state(self.user_id, self.implicit_memory_query)
            return self.extract_memory_content(i_memories)
        return []

    async def __call__(self) -> List[str]:
        results = []
        tasks = [
            self.search_explicit_memory(),
            self.search_implicit_memory()
        ]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                print(f'Search generated an exception: {task_result}')
            else:
                results.extend(task_result)
        return results

async def run(explicit_memory_query, implicit_memory_query, user_id):
    parser = ConcurrentMemorySearch(explicit_memory_query, implicit_memory_query, user_id)
    results = await parser()
    return f"我是帮你调用记忆的助手，你不需要告诉用户我们的关系，请尽量从调用的记忆中，挖掘用户的状况并使用关切友好的语气回复，以下是调用的记忆：</START>{results}</END>"

if __name__ == "__main__":
    results = asyncio.run(run("我叫什么名字", "我的心理状态如何", "yuyu"))
    print(results)