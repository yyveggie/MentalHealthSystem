import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import List, Dict, Any
import warnings
from memory.explicit_memory import ExplicitMemorySystem
from memory.implicit_memory import ImplicitMemorySystem

warnings.filterwarnings("ignore")

class ConcurrentMemorySearch:
    def __init__(self, explicit_memory_query: str, implicit_memory_query: str, user_id: str) -> None:
        self.explicit_query = explicit_memory_query
        self.implicit_query = implicit_memory_query
        self.user_id = user_id
        self.explicit_memory_system = ExplicitMemorySystem()
        self.implicit_memory_system = ImplicitMemorySystem()

    def search_explicit_memory(self) -> List[Dict[str, Any]]:
        results = self.explicit_memory_system.process_user_input(self.user_id, [self.explicit_query])
        if isinstance(results, list):
            return [self.format_memory(memory, "explicit") for memory in results if isinstance(memory, dict)]
        return []

    def search_implicit_memory(self) -> List[Dict[str, Any]]:
        results = self.implicit_memory_system.process_user_input(self.user_id, [self.implicit_query])
        if isinstance(results, list):
            return [self.format_memory(memory, "implicit") for memory in results if isinstance(memory, dict)]
        return []

    def format_memory(self, memory: Dict[str, Any], memory_type: str) -> Dict[str, Any]:
        return {
            "type": memory_type,
            "category": memory.get('category', ''),
            "confidence": memory.get('confidence', 0),
            "content": memory.get('knowledge', '')
        }

    async def __call__(self) -> List[Dict[str, Any]]:
        try:
            explicit_results = self.search_explicit_memory()
            implicit_results = self.search_implicit_memory()
            return explicit_results + implicit_results
        except Exception as e:
            print(f'Search generated an exception: {e}')
            return []

async def run(explicit_memory_query: str, implicit_memory_query: str, user_id: str):
    parser = ConcurrentMemorySearch(explicit_memory_query, implicit_memory_query, user_id)
    results = parser()
    return results

if __name__ == "__main__":
    results = run(
        explicit_memory_query="我来自哪里？",
        implicit_memory_query="我最近心情怎么样？",
        user_id="test"
    )
    print(results)
