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

import sys, os
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)
es = setup_logging()

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class Action(str, Enum):
    ADD = "添加"
    SEARCH = "搜索"
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
    inferences: Optional[List[MentalStateInference]] = Field(None, description="如果操作是'添加'，则为要添加的心理状态推断列表")
    search_query: Optional[str] = Field(None, description="如果操作是'搜索'，则为搜索查询")

    @field_validator("inferences")
    @classmethod
    def validate_inferences(cls, v, values):
        if values.data.get("action") == Action.ADD:
            if not v:
                raise ValueError("当操作为'添加'时，必须提供至少一个心理状态推断")
            for item in v:
                if item.category not in MentalStateCategory.__members__.values():
                    raise ValueError(f"无效的类别: {item.category}")
        return v

    @field_validator("search_query")
    @classmethod
    def validate_search_query(cls, v, values):
        if values.data.get("action") == Action.SEARCH and not v:
            raise ValueError("当操作为'搜索'时，必须提供搜索查询")
        return v

class MentalStateInferenceSystem:
    def __init__(self):
        logger.info("MentalStateInferenceSystem initialized")

    def _make_inference(self, query: str) -> MentalStateDecision:
        system_message = memory_prompt.implicit_prompt()
        try:
            logger.info(f"Making inference for query: {query[:50]}...")
            decision: MentalStateDecision = client.chat.completions.create(
                model=GPT4O,
                response_model=MentalStateDecision,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
            )
            logger.info(f"Inference made: Action={decision.action}")
            return decision
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            return MentalStateDecision(action=Action.DO_NOTHING)

    def process_query(self, user_id: str, query: str) -> Optional[str]:
        logger.info(f"Processing query for user {user_id}: {query[:50]}...")
        mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"{user_id}_mental_state",
                    "path": "./memory/implicit_db",
                }
            }
        })

        decision = self._make_inference(query)

        if decision.action == Action.ADD:
            added_inferences = []
            for inference_item in decision.inferences:
                mem0.add(inference_item.inference, user_id=user_id, metadata={
                    "category": inference_item.category.value,
                    "confidence": inference_item.confidence
                })
                added_inferences.append(f"{inference_item.inference}（类别：{inference_item.category.value}，置信度：{inference_item.confidence}）")
            logger.info(f"Added {len(added_inferences)} mental state inferences for user {user_id}")
            return f"记录的心理状态推断：\n" + "\n".join(added_inferences)
        elif decision.action == Action.SEARCH:
            logger.info(f"Searching for query: {decision.search_query} (user: {user_id})")
            search_results = mem0.search(decision.search_query, user_id=user_id)
            inference_values = [f"{result['memory']} (类别: {result['metadata']['category']}, 置信度: {result['metadata']['confidence']})" for result in search_results]
            logger.info(f"Found {len(inference_values)} search results for user {user_id}")
            return f"检索结果：{json.dumps(inference_values, ensure_ascii=False)}"
        else:  # DO_NOTHING
            logger.info(f"No implicit memory detected for user {user_id}")
            return "未检测到隐式记忆"

def infer_mental_state(user_id: str, query: str) -> Optional[str]:
    logger.info(f"Inferring mental state for user {user_id}")
    global mental_state_system
    if not hasattr(infer_mental_state, 'mental_state_system'):
        infer_mental_state.mental_state_system = MentalStateInferenceSystem()
    return infer_mental_state.mental_state_system.process_query(user_id, query)

def search_mental_state(user_id: str, query: str):
    logger.info(f"Searching mental state for user {user_id}: {query[:50]}...")
    mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./memory/implicit_db",
                }
            }
        })
    results = mem0.search(query=query, user_id=user_id)
    logger.info(f"Found {len(results)} search results for user {user_id}")
    return results

if __name__ == "__main__":
    user_id = "yuyu"
    query = "我感觉自己很难交到朋友，很没有耐心，也时常容易发脾气"
    result = infer_mental_state(user_id, query)
    if result:
        print(f"处理结果：\n{result}")
    else:
        print("推断患者心理状态时发生错误")