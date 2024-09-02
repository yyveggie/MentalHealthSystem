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
        pass

    def _make_decision(self, query: str) -> MemoryDecision:
        system_message = """你是一个专业的精神心理健康记录助手，负责记录患者的信息。你的任务是从患者的陈述中提取明确的信息，并决定如何处理这些信息。你有三个选择：

1. '添加'：如果患者提供了新的、具体的个人或医疗信息。
2. '搜索'：如果需要查找患者之前提供的信息。
3. '不做操作'：如果患者没有提供任何新的、具体的信息。

如果选择'添加'，请提供：
- 一个或多个记忆项，每个包含：
  * 患者陈述的具体信息。
  * 从以下选项中为信息选择一个最合适的类别：
    - 人口学信息（年龄、性别、职业、婚姻状况等）
    - 主诉（就诊的主要原因）
    - 现病史（当前问题的详细描述和发展过程）
    - 精神病史（既往精神疾病诊断和治疗）
    - 躯体疾病史（其他身体疾病）
    - 用药史（当前和既往的用药情况）
    - 物质使用史（酒精、烟草、药物滥用等）
    - 家族史（家族中的精神疾病和重大躯体疾病）
    - 发育史（儿童期发展、教育背景等）
    - 社会史（生活环境、人际关系、工作情况等）
    - 创伤史（重大生活事件和心理创伤）
    - 风险评估（自杀、自伤或暴力倾向）
    - 治疗史（既往的心理治疗和干预措施）
    - 支持系统（家庭支持、社会资源等）
    - 应对机制（患者处理压力和问题的方式）
    - 文化因素（可能影响症状表达或治疗的文化背景）
    - 优势和资源（患者的个人优势和可用资源）
    - 其他（不属于以上类别的重要信息）

如果选择'搜索'，请提供一个搜索查询来查找相关的既往信息。

重要提示：
- 只记录患者明确陈述的信息，不要进行推测或诊断。
- 专注于客观事实和患者的描述，而不是你的解释或分析。
- 保持中立，不要对患者的陈述做出判断。
- 如果患者的陈述不清晰或不够具体，选择'不做操作'。
- 如果一条信息可能属于多个类别，选择最相关或最具体的一个。

你明白了吗？深呼吸，做得好会给你一千美元。现在请仔细阅读并处理以下患者的陈述：
"""
        try:
            decision: MemoryDecision = client.chat.completions.create(
                model=GPT4O,
                response_model=MemoryDecision,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
            )
            return decision
        except Exception as e:
            print(f"处理输入时出错：{e}")
            return MemoryDecision(action=Action.DO_NOTHING)

    def process_query(self, user_id: str, query: str) -> Optional[str]:
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
            return f"记录的信息：\n" + "\n".join(added_memories)
        elif decision.action == Action.SEARCH:
            search_results = mem0.search(decision.search_query, user_id=user_id)
            memory_values = [result['memory'] for result in search_results]
            return f"检索结果：{json.dumps(memory_values, ensure_ascii=False)}"
        else:  # DO_NOTHING
            return "未记录新信息。"

def record_patient_info(user_id: str, query: str) -> Optional[str]:
    global patient_system
    if not hasattr(record_patient_info, 'patient_system'):
        record_patient_info.patient_system = PatientInformationSystem()
    return record_patient_info.patient_system.process_query(user_id, query)

def search_patient_info(user_id: str, query: str):
    mem0 = Memory.from_config({
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": user_id,
                    "path": "./memory/explicit_db",
                }
            }
        })
    return mem0.search(query=query, user_id=user_id)

def retrieve_all_memories(user_id: str):
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
        
    pprint(retrieve_all_memories(user_id))