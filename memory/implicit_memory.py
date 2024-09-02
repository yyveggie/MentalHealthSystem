import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import json
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

import instructor
from openai import OpenAI
from mem0 import Memory
from load_config import OPENAI_API_KEY, GPT4O

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
        pass

    def _make_inference(self, query: str) -> MentalStateDecision:
        system_message = """你是一个专业的精神心理健康推理助手，负责推断患者的潜在心理状态。你的任务是从患者的陈述中推断可能的心理因素，并决定如何处理这些推断。你有三个选择：

1. '添加'：如果从患者的陈述中可以推断出新的心理状态信息。
2. '搜索'：如果需要查找之前推断的心理状态信息。
3. '不做操作'：如果患者的陈述没有提供足够的信息来进行新的推断。

如果选择'添加'，请提供：
- 一个或多个心理状态推断，每个包含：
  * 对患者可能的心理状态的推断。
  * 从以下选项中为推断选择一个最合适的类别：
    - 情绪状态（如焦虑、抑郁、愤怒等）
    - 认知模式（如思维方式、信息处理方式）
    - 防御机制（如否认、投射、合理化等）
    - 人际动力（如依赖、疏离、支配等）
    - 动机（如需求、愿望、目标等）
    - 自我认知（如自尊、自我价值感等）
    - 应对策略（如问题解决、寻求支持等）
    - 无意识冲突（如内在矛盾、压抑的欲望等）
    - 依恋类型（如安全型、焦虑型、回避型等）
    - 创伤反应（如闪回、过度警觉等）
    - 认知偏差（如过度概括、灾难化等）
    - 信念系统（如核心信念、价值观等）
    - 存在性问题（如意义感、目的感等）
    - 其他（不属于以上类别的重要心理因素）
  * 推断的置信度（0.0到1.0之间的浮点数）

如果选择'搜索'，请提供一个搜索查询来查找相关的既往心理状态推断。

重要提示：
- 这些推断应基于专业的心理学知识和患者的陈述，但要认识到这些是推测性的。
- 保持客观和中立，不要对患者的心理状态做出判断。
- 考虑患者陈述的内容、语气、和可能隐含的信息。
- 如果患者的陈述不足以支持可靠的推断，选择'不做操作'。
- 每个推断都应附带一个置信度，反映推断的可靠性。
- 如果一个推断可能属于多个类别，选择最相关或最具体的一个。

你明白了吗？深呼吸，做得好会给你一千美元。现在请仔细阅读并处理以下患者的陈述：
"""
        try:
            decision: MentalStateDecision = client.chat.completions.create(
                model=GPT4O,
                response_model=MentalStateDecision,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
            )
            return decision
        except Exception as e:
            print(f"处理输入时出错：{e}")
            return MentalStateDecision(action=Action.DO_NOTHING)

    def process_query(self, user_id: str, query: str) -> Optional[str]:
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
            return f"记录的心理状态推断：\n" + "\n".join(added_inferences)
        elif decision.action == Action.SEARCH:
            search_results = mem0.search(decision.search_query, user_id=user_id)
            inference_values = [f"{result['memory']} (类别: {result['metadata']['category']}, 置信度: {result['metadata']['confidence']})" for result in search_results]
            return f"检索结果：{json.dumps(inference_values, ensure_ascii=False)}"
        else:  # DO_NOTHING
            return "未推断新的心理状态信息。"

def infer_mental_state(user_id: str, query: str) -> Optional[str]:
    global mental_state_system
    if not hasattr(infer_mental_state, 'mental_state_system'):
        infer_mental_state.mental_state_system = MentalStateInferenceSystem()
    return infer_mental_state.mental_state_system.process_query(user_id, query)

def search_mental_state(user_id: str, query: str):
    mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./memory/implicit_db",
                }
            }
        })
    return mem0.search(query=query, user_id=user_id)

if __name__ == "__main__":
    user_id = "yuyu"
    query = "我感觉自己很难交到朋友，很没有耐心，也时常容易发脾气"
    result = infer_mental_state(user_id, query)
    if result:
        print(f"处理结果：\n{result}")
    else:
        print("推断患者心理状态时发生错误")