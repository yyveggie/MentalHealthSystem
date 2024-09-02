import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from memory import explicit_memory, implicit_memory
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    print(f"Final results: {results}")
    return results

if __name__ == "__main__":
    results = run("我叫什么名字", "我的心理状态如何", "yuyu")
    print(f"Main function results: {results}")