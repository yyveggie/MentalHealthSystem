import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
import asyncio
import instructor
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from openai import AsyncOpenAI
from mem0 import Memory

from prompts import memory_prompt
from load_config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, CHAT_MODEL, HOST, API_KEY

import logging
from logging_config import setup_logging

logger = logging.getLogger(__name__)
es = setup_logging()

client = instructor.from_openai(
    AsyncOpenAI(
        base_url=HOST + "/v1",
        api_key=API_KEY,  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

class Action(str, Enum):
    ADD = "添加"
    SEARCH = "搜索"
    DO_NOTHING = "不做操作"

class Category(str, Enum):
    DEMOGRAPHIC_INFO = "人口学信息"
    CHIEF_COMPLAINT = "主诉"
    HISTORY_PRESENT_ILLNESS = "现病史"
    PSYCHIATRIC_HISTORY = "精神病史"
    MEDICAL_HISTORY = "躯体疾病史"
    MEDICATION_HISTORY = "用药史"
    SUBSTANCE_USE = "物质使用史"
    FAMILY_HISTORY = "家族史"
    DEVELOPMENTAL_HISTORY = "发育史"
    SOCIAL_HISTORY = "社会史"
    TRAUMA_HISTORY = "创伤史"
    RISK_ASSESSMENT = "风险评估"
    TREATMENT_HISTORY = "治疗史"
    SUPPORT_SYSTEM = "支持系统"
    COPING_MECHANISMS = "应对机制"
    CULTURAL_FACTORS = "文化因素"
    STRENGTHS_RESOURCES = "优势和资源"
    OTHER = "其他"

class MemoryItem(BaseModel):
    memory: str
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in Category.__members__.values():
            raise ValueError(f"无效的类别: {v}")
        return v

class MemoryDecision(BaseModel):
    action: Action
    memories: Optional[List[MemoryItem]] = Field(None, description="如果操作是'添加'，则为要添加的记忆列表")
    search_query: Optional[str] = Field(None, description="如果操作是'搜索'，则为搜索查询")

    @field_validator("memories")
    @classmethod
    def validate_memories(cls, v, values):
        if values.data.get("action") == Action.ADD:
            if not v:
                raise ValueError("当操作为'添加'时，必须提供至少一个记忆项")
            for item in v:
                if item.category not in Category.__members__.values():
                    raise ValueError(f"无效的类别: {item.category}")
        return v

    @field_validator("search_query")
    @classmethod
    def validate_search_query(cls, v, values):
        if values.data.get("action") == Action.SEARCH and not v:
            raise ValueError("当操作为'搜索'时，必须提供搜索查询")
        return v

class PatientInformationSystem:
    def __init__(self, user_id):
        logger.info("PatientInformationSystem initialized")
        self.user_id = user_id
        self.mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./database/memory/explicit",
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": CHAT_MODEL,
                    "temperature": 0,
                    "max_tokens": 8000,
                    "ollama_base_url": HOST,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": EMBEDDING_MODEL,
                    "embedding_dims": EMBEDDING_DIMENSION,
                    "ollama_base_url": HOST
                }
            }
        })

    async def _make_decision(self, query: str) -> MemoryDecision:
        system_message = memory_prompt.explicit_prompt()
        try:
            logger.info(f"Making decision for query: {query[:50]}...")
            decision: MemoryDecision = await client.chat.completions.create(
                model="qwen2.5:32b",
                response_model=MemoryDecision,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
            )
            logger.info(f"Decision made: Action={decision.action}")
            return decision
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            return MemoryDecision(action=Action.DO_NOTHING)

    async def process_query(self, user_id: str, query: str) -> Optional[str]:
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")

        decision = await self._make_decision(query)

        if decision.action == Action.ADD:
            added_memories = []
            for memory_item in decision.memories:
                memory_content = json.dumps({
                    "memory": memory_item.memory,
                    "category": memory_item.category.value,
                    "confidence": memory_item.confidence
                })
                await asyncio.to_thread(
                    self.mem0.add,
                    memory_content,
                    user_id=user_id,
                    metadata={
                        "category": memory_item.category.value,
                        "confidence": memory_item.confidence
                    }
                )
                added_memories.append(f"{memory_item.memory}（类别：{memory_item.category.value}，置信度：{memory_item.confidence}）")
            logger.info(f"Added {len(added_memories)} memories for user {user_id}")
            return f"记录的信息：\n" + "\n".join(added_memories)
        elif decision.action == Action.SEARCH:
            logger.info(f"Searching for query: {decision.search_query} (user: {user_id})")
            search_results = await asyncio.to_thread(self.mem0.search, decision.search_query, user_id=user_id)
            memory_values = [f"{json.loads(result['memory'])['memory']} (类别: {result['metadata']['category']}, 置信度: {result['metadata']['confidence']})" for result in search_results]
            logger.info(f"Found {len(memory_values)} search results for user {user_id}")
            return f"检索结果：{json.dumps(memory_values, ensure_ascii=False)}"
        else:  # DO_NOTHING
            logger.info(f"No explicit memory detected for user {user_id}")
            return "未检测到显式记忆"

async def record_patient_info(user_id: str, query: str) -> Optional[str]:
    logger.info(f"Recording patient info for user {user_id}")
    patient_system = PatientInformationSystem(user_id=user_id)
    return await patient_system.process_query(user_id, query)

async def search_patient_info(user_id: str, query: str):
    logger.info(f"Searching patient info for user {user_id}: {query[:50]}...")
    patient_system = PatientInformationSystem(user_id=user_id)
    results = await asyncio.to_thread(patient_system.mem0.search, query=query, user_id=user_id)
    logger.info(f"Found {len(results)} search results for user {user_id}")
    return results

async def retrieve_all_memories(user_id: str):
    logger.info(f"Retrieving all memories for user {user_id}")
    patient_system = PatientInformationSystem(user_id=user_id)
    all_memories = await asyncio.to_thread(patient_system.mem0.get_all, user_id=user_id)
    logger.info(f"Retrieved {len(all_memories)} memories for user {user_id}")
    return all_memories

async def main():
    from pprint import pprint
    user_id = "yuyu"
    query = "我改名了，不叫cyy，现在叫cxx，我来自于绍兴市，目前在上海念书。"
    result = await record_patient_info(user_id, query)
    if result:
        print(f"处理结果：\n{result}")
    else:
        print("处理患者信息时发生错误")
        
    all_memories = await retrieve_all_memories(user_id)
    pprint(all_memories)

if __name__ == "__main__":
    asyncio.run(main())