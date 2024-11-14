import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import json
from enum import Enum
from typing import Dict, List, TypedDict, Sequence, Optional, Any
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.prebuilt import ToolExecutor
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

from prompts import memory_prompt
from utils.mongodb_patient_info_system import MongoDBPatientInfoSystem
from load_config import CHAT_MODEL, API_KEY, MONGODB_HOST, MONGODB_PORT

from pymongo import MongoClient
from typing import Dict, List, Any

class MongoDBPatientInfoSystem:
    def __init__(self, connection_string: str):
        self.client = MongoClient(connection_string)

    def get_user_db(self, user_id: str):
        return self.client[user_id]

    def get_category_collection(self, user_id: str, category: str):
        db = self.get_user_db(user_id)
        return db[category]

    def add_memory(self, user_id: str, category: str, memory: Dict[str, Any]):
        collection = self.get_category_collection(user_id, category)
        result = collection.insert_one(memory)
        return result.inserted_id

    def update_memory(self, user_id: str, category: str, memory_id: str, updated_memory: Dict[str, Any]):
        collection = self.get_category_collection(user_id, category)
        result = collection.update_one({"_id": memory_id}, {"$set": updated_memory})
        return result.modified_count

    def delete_memory(self, user_id: str, category: str, memory_id: str):
        collection = self.get_category_collection(user_id, category)
        result = collection.delete_one({"_id": memory_id})
        return result.deleted_count

    def get_memories(self, user_id: str, category: str) -> List[Dict[str, Any]]:
        collection = self.get_category_collection(user_id, category)
        return list(collection.find())

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

class Action(str, Enum):
    Create = "创建"
    Update = "更新"
    Delete = "删除"

class AddPatientKnowledge(BaseModel):
    knowledge: str = Field(
        ...,
        description="要保存的患者知识的简洁表述。格式：[类别]: [详细信息]",
    )
    knowledge_old: Optional[str] = Field(
        None,  
        description="如果是更新或删除记录，需要修改的完整、准确的原始短语",
    )
    category: Category = Field(
        ...,
        description="此知识所属的类别" 
    )
    confidence: float = Field(
        ...,
        description="此信息的置信度，从 0.0 到 1.0",
    )
    action: Action = Field(
        ...,
        description="此知识是添加新记录、更新记录还是删除记录",
    )

def modify_patient_knowledge(
    knowledge: str,
    category: str, 
    confidence: float,
    action: str,
    knowledge_old: str = "",
) -> dict:
    return {"status": "Success", "message": "Patient knowledge modified"} 

tool_modify_patient_knowledge = StructuredTool.from_function(
    func=modify_patient_knowledge,
    name="Patient_Knowledge_Modifier",
    description="Add, update, or delete a bit of patient knowledge", 
    args_schema=AddPatientKnowledge,
)

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    memories: Dict[str, List[str]]
    contains_information: Optional[str]
    
def evaluate_sentinel_response(response_content: str) -> bool:
    positive_indicators = ["TRUE", "是的", "包含", "有价值"]
    negative_indicators = ["FALSE", "没有", "不包含", "无价值"]
    
    response_lower = response_content.lower()
    
    positive_score = sum(1 for indicator in positive_indicators if indicator.lower() in response_lower)
    negative_score = sum(1 for indicator in negative_indicators if indicator.lower() in response_lower)
    
    return positive_score > negative_score

def parse_sentinel_response(response_content: str) -> Optional[str]:
    response_lower = response_content.lower().strip()
    if "false" in response_lower or "不做操作" in response_lower or "无需新增或修改" in response_lower:
        return None
    for category in Category:
        if category.value in response_content:
            return category.value
    return None

def call_sentinel(state):
    messages = state["messages"]
    response = state["sentinel_runnable"].invoke(messages)
    category = parse_sentinel_response(response.content)
    return {"contains_information": category if category else "False"}

def should_continue(state):
    if state["contains_information"] == "False":
        return "end"
    else:
        return "continue"

def call_knowledge_master(state):
    messages = state["messages"]
    category = state["contains_information"]
    memories = state["memories"].get(category, [])
    response = state["knowledge_master_runnable"].invoke(
        {"messages": messages, "memories": memories}
    )
    return {"messages": messages + [response]}

def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    new_memories = state["memories"].copy()

    for tool_call in last_message.additional_kwargs.get("tool_calls", []):
        action = ToolInvocation(
            tool=tool_call["function"]["name"],
            tool_input=json.loads(tool_call["function"]["arguments"]),
            id=tool_call["id"],
        )

        response_dict = state["tool_executor"].invoke(action)

        knowledge_info = action.tool_input
        category = knowledge_info.get("category")
        if category:
            if category not in new_memories:
                new_memories[category] = []
            new_memories[category].append(str(knowledge_info))

    return {"messages": messages, "memories": new_memories}

class ExplicitMemorySystem:
    def __init__(self):
        self.db_system = MongoDBPatientInfoSystem(f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}/")
        self.agent_tools = [tool_modify_patient_knowledge]
        self.tool_executor = ToolExecutor(self.agent_tools)

        sentinel_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(memory_prompt.explicit_initial_sentinel_prompt()),
            MessagesPlaceholder(variable_name="messages"),
        ])
        knowledge_master_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(memory_prompt.explicit_initial_knowledge_master_prompt()),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessagePromptTemplate.from_template("现有的记忆：{memories}"),
        ])

        tools = [convert_to_openai_function(t) for t in self.agent_tools]

        self.sentinel_runnable = sentinel_prompt | ChatOpenAI(
            temperature=0,
            model=CHAT_MODEL,
            api_key=API_KEY,
            # base_url=HOST + "/v1"
        )
        self.knowledge_master_runnable = knowledge_master_prompt | ChatOpenAI(
            temperature=0.5,
            model=CHAT_MODEL,
            api_key=API_KEY,
            # base_url=HOST + "/v1"
        ).bind_tools(tools)

    def process_user_input(self, user_id: str, conversation_history: List[str]):
        messages = [HumanMessage(content=msg) for msg in conversation_history]
        
        response = self.sentinel_runnable.invoke({"messages": messages})
        contains_information = parse_sentinel_response(response.content)

        if contains_information:
            category = contains_information
            memories = self.db_system.get_memories(user_id, category)
            response = self.knowledge_master_runnable.invoke(
                {"messages": messages, "memories": memories}
            )
            messages.append(response)
            last_message = messages[-1]

            if "tool_calls" in last_message.additional_kwargs:
                new_memories = self.process_tool_calls(user_id, category, last_message.additional_kwargs["tool_calls"])
                return new_memories if new_memories else "无显式记忆记录"
            else:
                return "无显式记忆记录"
        return "无显式记忆记录"
    
    def process_tool_calls(self, user_id: str, category: str, tool_calls):
        new_memories = []
        for tool_call in tool_calls:
            action = ToolInvocation(
                tool=tool_call["function"]["name"],
                tool_input=json.loads(tool_call["function"]["arguments"]),
                id=tool_call["id"],
            )
            response_dict = self.tool_executor.invoke(action)
            
            if isinstance(action.tool_input, dict):
                new_memories.append(action.tool_input)
                self.handle_tool_response(user_id, response_dict, action.tool_input)
        return new_memories

    def handle_tool_response(self, user_id: str, response_dict, tool_input):
        if response_dict["status"] == "Success":
            action = tool_input.get('action')
            category = tool_input.get('category')
            if action == Action.Create:
                self.db_system.add_memory(user_id, category, tool_input)
                print(f"成功创建新知识: {tool_input['knowledge']}")
            elif action == Action.Update:
                if 'knowledge_old' in tool_input:
                    self.update_existing_memory(user_id, category, tool_input)
                else:
                    print(f"警告: 尝试更新不存在的知识。将其作为新知识添加。")
                    self.db_system.add_memory(user_id, category, tool_input)
            elif action == Action.Delete:
                if 'knowledge_old' in tool_input:
                    self.delete_existing_memory(user_id, category, tool_input)
                else:
                    print(f"警告: 尝试删除不存在的知识。忽略此操作。")
        else:
            print(f"处理知识时出错: {response_dict['message']}")

    def update_existing_memory(self, user_id: str, category: str, memory: Dict[str, Any]):
        memories = self.db_system.get_memories(user_id, category)
        for existing_memory in memories:
            if existing_memory['knowledge'] == memory.get('knowledge_old'):
                self.db_system.update_memory(user_id, category, existing_memory['_id'], memory)
                print(f"成功更新知识: {memory['knowledge']}")
                return
        
        print(f"警告: 未找到要更新的知识。将其作为新知识添加。")
        self.db_system.add_memory(user_id, category, memory)

    def delete_existing_memory(self, user_id: str, category: str, memory: Dict[str, Any]):
        memories = self.db_system.get_memories(user_id, category)
        for existing_memory in memories:
            if existing_memory['knowledge'] == memory.get('knowledge_old'):
                self.db_system.delete_memory(user_id, category, existing_memory['_id'])
                print(f"成功删除知识。")
                return
        print(f"警告: 未找到要删除的知识。")
            

if __name__ == "__main__":
    user_id = "test"
    knowledge_base = ExplicitMemorySystem()
    user_input = "我在上海财经大学念书"
    print(knowledge_base.process_user_input(user_id, [user_input]))

graph = StateGraph(AgentState)

graph.add_node("sentinel", call_sentinel)
graph.add_node("knowledge_master", call_knowledge_master)
graph.add_node("action", call_tool)

graph.set_entry_point("sentinel")

graph.add_conditional_edges(
    "sentinel",
    should_continue,
    {
        "continue": "knowledge_master",
        "end": END,
    },
)
graph.add_edge("knowledge_master", "action")
graph.add_edge("action", END)

app = graph.compile()