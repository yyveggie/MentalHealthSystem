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
    ADD = "新增或修正"
    DO_NOTHING = "不做操作"

class MentalStateCategory(str, Enum):
    EMOTIONAL_STATE = "情绪状态"
    COGNITIVE_PATTERNS = "认知模式"
    DEFENSE_MECHANISMS = "防御机制"
    INTERPERSONAL_DYNAMICS = "人际动力"
    MOTIVATION = "动机"
    SELF_PERCEPTION = "自我认知"
    COPING_STRATEGIES = "应对策略"
    UNCONSCIOUS_CONFLICTS = "无意识冲突"
    ATTACHMENT_STYLE = "依恋类型"
    TRAUMA_RESPONSE = "创伤反应"
    COGNITIVE_BIASES = "认知偏差"
    BELIEF_SYSTEMS = "信念系统"
    EXISTENTIAL_CONCERNS = "存在性问题"
    OTHER = "其他"

class MentalStateInference(BaseModel):
    inference: str
    category: MentalStateCategory
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in MentalStateCategory.__members__.values():
            raise ValueError(f"无效的心理状态类别: {v}")
        return v

class MentalStateDecision(BaseModel):
    action: Action
    inferences: Optional[List[MentalStateInference]] = Field(None, description="如果操作是'新增或修正'，则为要新增或修正的心理状态推断。可以是多条推断的列表")

    @field_validator("inferences")
    @classmethod
    def validate_inferences(cls, v, values):
        if values.data.get("action") == Action.ADD:
            if not v:
                raise ValueError("当操作为'新增或修正'时，必须提供至少一个心理状态推断")
            for item in v:
                if item.category not in MentalStateCategory.__members__.values():
                    raise ValueError(f"无效的类别: {item.category}")
        return v

class MentalStateInferenceSystem:
    def __init__(self, user_id):
        logger.info("MentalStateInferenceSystem initialized")
        self.user_id = user_id
        self.mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./database/memory/implicit",
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

    async def _make_decision(self, query: str) -> MentalStateDecision:
        system_message = memory_prompt.implicit_prompt()
        try:
            logger.info(f"Making decision for query: {query[:50]}...")
            decision: MentalStateDecision = await client.chat.completions.create(
                model=CHAT_MODEL,
                response_model=MentalStateDecision,
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
            return MentalStateDecision(action=Action.DO_NOTHING)

    async def process_query(self, user_id: str, query: str) -> Optional[str]:
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")

        decision = await self._make_decision(query)

        if decision.action == Action.ADD:
            added_inferences = []
            for inference_item in decision.inferences:
                memory_content = json.dumps({
                    "inference": inference_item.inference,
                    "category": inference_item.category.value,
                    "confidence": inference_item.confidence
                })
                await asyncio.to_thread(
                    self.mem0.add,
                    memory_content,
                    user_id=user_id,
                    metadata={
                        "category": inference_item.category.value,
                        "confidence": inference_item.confidence
                    }
                )
                added_inferences.append(f"{inference_item.inference}（类别：{inference_item.category.value}，置信度：{inference_item.confidence}）")
            logger.info(f"Added {len(added_inferences)} mental state inferences for user {user_id}")
            return f"记录的心理状态推断：\n" + "\n".join(added_inferences)
        else:
            logger.info(f"No implicit memory detected for user {user_id}")
            return "未检测到隐式记忆"

async def infer_mental_state(user_id: str, query: str) -> Optional[str]:
    logger.info(f"Inferring mental state for user {user_id}")
    mental_state_system = MentalStateInferenceSystem(user_id=user_id)
    return await mental_state_system.process_query(user_id, query)

async def search_mental_state(user_id: str, query: str):
    logger.info(f"Searching mental state for user {user_id}: {query[:50]}...")
    mental_state_system = MentalStateInferenceSystem(user_id=user_id)
    results = await asyncio.to_thread(mental_state_system.mem0.search, query=query, user_id=user_id)
    logger.info(f"Found {len(results)} search results for user {user_id}")
    return results

async def retrieve_all_mental_states(user_id: str):
    logger.info(f"Retrieving all mental states for user {user_id}")
    mental_state_system = MentalStateInferenceSystem(user_id=user_id)
    all_mental_states = await asyncio.to_thread(mental_state_system.mem0.get_all, user_id=user_id)
    logger.info(f"Retrieved {len(all_mental_states)} mental states for user {user_id}")
    return all_mental_states

async def main():
    from pprint import pprint
    user_id = "yuyu"
    query = "我感觉自己很难交到朋友，很没有耐心，也时常容易发脾气"
    result = await infer_mental_state(user_id, query)
    if result:
        print(f"处理结果：\n{result}")
    else:
        print("推断患者心理状态时发生错误")
        
    all_mental_states = await retrieve_all_mental_states(user_id)
    pprint(all_mental_states)

if __name__ == "__main__":
    asyncio.run(main())