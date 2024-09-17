import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from memory import explicit_memory, implicit_memory
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore")

class ConcurrentMemorySearch:
    def __init__(self, explicit_memory_query, implicit_memory_query, user_id) -> None:
        self.explicit_memory_query = explicit_memory_query
        self.implicit_memory_query = implicit_memory_query
        self.user_id = user_id

    def extract_memory_content(self, memories: List[Dict[str, Any]]) -> List[str]:
        return [memory.get('memory', '') for memory in memories if 'memory' in memory]

    def search_explicit_memory(self) -> List[str]:
        if self.explicit_memory_query:
            e_memories = explicit_memory.search_patient_info(self.user_id, self.explicit_memory_query)
            return self.extract_memory_content(e_memories)
        return []

    def search_implicit_memory(self) -> List[str]:
        if self.implicit_memory_query:
            i_memories = implicit_memory.search_mental_state(self.user_id, self.implicit_memory_query)
            return self.extract_memory_content(i_memories)
        return []

    def __call__(self) -> List[str]:
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_search = {
                executor.submit(self.search_explicit_memory): "explicit",
                executor.submit(self.search_implicit_memory): "implicit"
            }
            for future in as_completed(future_to_search):
                search_type = future_to_search[future]
                try:
                    data = future.result()
                    results.extend(data)
                except Exception as exc:
                    print(f'{search_type} search generated an exception: {exc}')
        return results

def run(explicit_memory_query, implicit_memory_query, user_id):
    parser = ConcurrentMemorySearch(explicit_memory_query, implicit_memory_query, user_id)
    results = parser()
    return f"我是帮你调用记忆的助手，你不需要告诉用户我们的关系，请尽量从调用的记忆中，挖掘用户的状况并使用关切友好的语气回复，以下是调用的记忆：</START>{results}</END>"

if __name__ == "__main__":
    results = run("我叫什么名字", "我的心理状态如何", "yuyu")
    print(f"Main function results: {results}")