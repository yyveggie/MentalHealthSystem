import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import json
import instructor
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from mem0 import Memory

from prompts import memory_prompt
from load_config import OPENAI_API_KEY, GPT4O

import logging
from logging_config import setup_logging

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

logger = logging.getLogger(__name__)
es = setup_logging()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class Action(str, Enum):
    ADD = "添加"
    SEARCH = "搜索"
    DO_NOTHING = "不做操作"

class Category(str, Enum):
    DEMOGRAPHIC_INFO = "人口学信息"  # 包括年龄、性别、职业、婚姻状况等
    CHIEF_COMPLAINT = "主诉"  # 患者就诊的主要原因
    HISTORY_PRESENT_ILLNESS = "现病史"  # 当前问题的详细描述和发展过程
    PSYCHIATRIC_HISTORY = "精神病史"  # 既往精神疾病诊断和治疗
    MEDICAL_HISTORY = "躯体疾病史"  # 其他身体疾病史
    MEDICATION_HISTORY = "用药史"  # 当前和既往的用药情况
    SUBSTANCE_USE = "物质使用史"  # 包括酒精、烟草、药物滥用等
    FAMILY_HISTORY = "家族史"  # 家族中的精神疾病和重大躯体疾病
    DEVELOPMENTAL_HISTORY = "发育史"  # 儿童期发展、教育背景等
    SOCIAL_HISTORY = "社会史"  # 生活环境、人际关系、工作情况等
    TRAUMA_HISTORY = "创伤史"  # 重大生活事件和心理创伤
    RISK_ASSESSMENT = "风险评估"  # 自杀、自伤或暴力倾向
    TREATMENT_HISTORY = "治疗史"  # 既往的心理治疗和干预措施
    SUPPORT_SYSTEM = "支持系统"  # 家庭支持、社会资源等
    COPING_MECHANISMS = "应对机制"  # 患者处理压力和问题的方式
    CULTURAL_FACTORS = "文化因素"  # 可能影响症状表达或治疗的文化背景
    STRENGTHS_RESOURCES = "优势和资源"  # 患者的个人优势和可用资源
    OTHER = "其他"  # 不属于以上类别的重要信息

class MemoryItem(BaseModel):
    memory: str
    category: Category

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
    def __init__(self):
        logger.info("PatientInformationSystem initialized")

    def _make_decision(self, query: str) -> MemoryDecision:
        system_message = memory_prompt.explicit_prompt()
        try:
            logger.info(f"Making decision for query: {query[:50]}...")  # Log only first 50 chars for brevity
            decision: MemoryDecision = client.chat.completions.create(
                model=GPT4O,
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

    def process_query(self, user_id: str, query: str) -> Optional[str]:
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")
        mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./memory/explicit_db",
                }
            }
        })

        decision = self._make_decision(query)

        if decision.action == Action.ADD:
            added_memories = []
            for memory_item in decision.memories:
                mem0.add(memory_item.memory, user_id=user_id, metadata={"category": memory_item.category.value})
                added_memories.append(f"{memory_item.memory}（类别：{memory_item.category.value}）")
            logger.info(f"Added {len(added_memories)} memories for user {user_id}")
            return f"记录的信息：\n" + "\n".join(added_memories)
        elif decision.action == Action.SEARCH:
            logger.info(f"Searching for query: {decision.search_query} (user: {user_id})")
            search_results = mem0.search(decision.search_query, user_id=user_id)
            memory_values = [result['memory'] for result in search_results]
            logger.info(f"Found {len(memory_values)} search results for user {user_id}")
            return f"检索结果：{json.dumps(memory_values, ensure_ascii=False)}"
        else:  # DO_NOTHING
            logger.info(f"No explicit memory detected for user {user_id}")
            return "未检测到显式记忆"

def record_patient_info(user_id: str, query: str) -> Optional[str]:
    logger.info(f"Recording patient info for user {user_id}")
    global patient_system
    if not hasattr(record_patient_info, 'patient_system'):
        record_patient_info.patient_system = PatientInformationSystem()
    return record_patient_info.patient_system.process_query(user_id, query)

def search_patient_info(user_id: str, query: str):
    logger.info(f"Searching patient info for user {user_id}: {query[:50]}...")
    mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./memory/explicit_db",
                }
            }
        })
    results = mem0.search(query=query, user_id=user_id)
    logger.info(f"Found {len(results)} search results for user {user_id}")
    return results

def retrieve_all_memories(user_id: str):
    logger.info(f"Retrieving all memories for user {user_id}")
    mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./memory/explicit_db",
                }
            }
        })
    all_memories = mem0.get_all(user_id=user_id)
    logger.info(f"Retrieved {len(all_memories)} memories for user {user_id}")
    return all_memories

if __name__ == "__main__":
    from pprint import pprint
    user_id = "yuyu"
    query = "我叫cyy，我来自于绍兴市，目前在上海念书。"
    result = record_patient_info(user_id, query)
    if result:
        print(f"处理结果：\n{result}")
    else:
        print("处理患者信息时发生错误")
        
    all_memories = retrieve_all_memories(user_id)
    pprint(all_memories)