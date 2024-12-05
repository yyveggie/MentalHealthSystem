import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from enum import Enum
from datetime import datetime
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
from memory.memory_retrieve import MemoryRetrievalSystem
from load_config import CHAT_MODEL, API_KEY, MONGODB_HOST, MONGODB_PORT
from pymongo import MongoClient

class MemoryType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"

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
        memory['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory['is_latest'] = True
        result = collection.insert_one(memory)
        return result.inserted_id

    def update_memory(self, user_id: str, category: str, memory_id: str, updated_memory: Dict[str, Any]):
        collection = self.get_category_collection(user_id, category)
        original_memory = collection.find_one({"_id": memory_id})
        if not original_memory:
            return None
            
        collection.update_one(
            {"_id": memory_id},
            {"$set": {"is_latest": False}}
        )

        updated_memory['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_memory['previous_version_id'] = str(memory_id)
        updated_memory['is_latest'] = True
        
        result = collection.insert_one(updated_memory)
        return result.inserted_id

    def get_memories(self, user_id: str, category: str, include_history: bool = False):
        collection = self.get_category_collection(user_id, category)
        if include_history:
            return list(collection.find().sort("timestamp", -1))
        else:
            query = {
                "$or": [
                    {"action": "创建"},
                    {"$and": [{"action": "更新"}, {"is_latest": True}]}
                ]
            }
            return list(collection.find(query))

    def get_memory_history(self, user_id: str, category: str, memory_id: str) -> List[Dict[str, Any]]:
        collection = self.get_category_collection(user_id, category)
        history = []
        current_id = memory_id
        
        while current_id:
            memory = collection.find_one({"_id": current_id})
            if memory:
                history.append(memory)
                current_id = memory.get('previous_version_id')
            else:
                break
                
        return history

class ExplicitCategory(str, Enum):
    DEMOGRAPHIC_INFO = "人口学信息"
    CHIEF_COMPLAINT = "主诉"
    HISTORY_PRESENT_ILLNESS = "现病史"
    MEDICATION_HISTORY = "用药史"
    SUBSTANCE_USE = "物质使用史"
    FAMILY_HISTORY = "家族史"
    SOCIAL_HISTORY = "社会史"
    TRAUMA_HISTORY = "创伤史"
    TREATMENT_HISTORY = "治疗史"

class ImplicitCategory(str, Enum):
    EMOTIONAL = "情绪体验"
    BEHAVIORAL = "行为模式"
    COGNITIVE = "认知特征"
    HISTORY = "历史信息"
    PERSONALITY = "人格特质"

class Action(str, Enum):
    Create = "创建"
    Update = "更新"

class AddPatientKnowledge(BaseModel):
    knowledge: str = Field(
        ...,
        description="要保存的患者知识的简洁表述。格式：[类别]: [详细信息]",
    )
    knowledge_old: Optional[str] = Field(
        None,  
        description="如果是更新记录，需要修改的完整、准确的原始短语",
    )
    category: str = Field(
        ...,
        description="此知识所属的类别" 
    )
    confidence: float = Field(
        ...,
        description="此信息的置信度，从 0.0 到 1.0",
    )
    action: Action = Field(
        ...,
        description="此知识是添加新记录还是更新记录",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="记录的最后更新时间"
    )
    previous_version_id: Optional[str] = Field(
        None,
        description="指向前一个版本记忆的ID"
    )
    is_latest: bool = Field(
        default=True,
        description="是否是最新版本"
    )

def modify_patient_knowledge(
    knowledge: str,
    category: str, 
    confidence: float,
    action: str,
    knowledge_old: str = "",
    timestamp: str = "",
    previous_version_id: Optional[str] = None,
    is_latest: bool = True,
) -> dict:
    return {"status": "Success", "message": "Patient knowledge modified"}

tool_modify_patient_knowledge = StructuredTool.from_function(
    func=modify_patient_knowledge,
    name="Patient_Knowledge_Modifier",
    description="Add, or update a bit of patient knowledge", 
    args_schema=AddPatientKnowledge,
)

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    memories: Dict[str, List[str]]
    contains_information: Optional[str]

class BaseMemorySystem:
    def __init__(self, memory_type: MemoryType):
        self.memory_type = memory_type
        self.db_system = MongoDBPatientInfoSystem(f"mongodb://{MONGODB_HOST}:{MONGODB_PORT}/")
        self.agent_tools = [tool_modify_patient_knowledge]
        self.tool_executor = ToolExecutor(self.agent_tools)
        self.category_enum = ExplicitCategory if memory_type == MemoryType.EXPLICIT else ImplicitCategory
        
        sentinel_template = (
            memory_prompt.explicit_initial_sentinel_prompt()
            if memory_type == MemoryType.EXPLICIT
            else memory_prompt.implicit_initial_sentinel_prompt()
        )
        knowledge_master_template = (
            memory_prompt.explicit_initial_knowledge_master_prompt()
            if memory_type == MemoryType.EXPLICIT
            else memory_prompt.implicit_initial_knowledge_master_prompt()
        )

        sentinel_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sentinel_template),
            MessagesPlaceholder(variable_name="messages"),
        ])
        knowledge_master_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(knowledge_master_template),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessagePromptTemplate.from_template("现有的记忆：{memories}"),
        ])

        tools = [convert_to_openai_function(t) for t in self.agent_tools]

        self.sentinel_runnable = sentinel_prompt | ChatOpenAI(
            temperature=0,
            model=CHAT_MODEL,
            api_key=API_KEY,
        )
        self.knowledge_master_runnable = knowledge_master_prompt | ChatOpenAI(
            temperature=0.5,
            model=CHAT_MODEL,
            api_key=API_KEY,
        ).bind_tools(tools)

    def process_user_input(self, user_id: str, conversation_history: List[str]):
        messages = [HumanMessage(content=msg) for msg in conversation_history]
        
        print("开始处理用户输入...")
        response = self.sentinel_runnable.invoke({"messages": messages})
        print(f"Sentinel响应: {response.content}")
        contains_information = parse_sentinel_response(response.content, self.memory_type)
        print(f"解析结果: {contains_information}")

        if contains_information:
            retrieval_system = MemoryRetrievalSystem()
            categories = list(self.category_enum._value2member_map_.keys())
            memories = retrieval_system.retrieve_memories_by_categories(categories)
            memories = json.dumps(memories, ensure_ascii=False, default=str)
            print(memories)

            knowledge_master_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    (memory_prompt.explicit_initial_knowledge_master_prompt() 
                     if self.memory_type == MemoryType.EXPLICIT 
                     else memory_prompt.implicit_initial_knowledge_master_prompt()) + "\n" +
                    f"当前类别：{contains_information}\n" +
                    "已有记忆：{memories}"
                ),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            self.knowledge_master_runnable = knowledge_master_prompt | ChatOpenAI(
                temperature=0.5,
                model=CHAT_MODEL,
                api_key=API_KEY,
            ).bind_tools([convert_to_openai_function(t) for t in self.agent_tools])
            
            response = self.knowledge_master_runnable.invoke({
                "messages": messages,
                "memories": memories if memories else "（该类别暂无记忆）"
            })
            messages.append(response)
            last_message = messages[-1]

            if "tool_calls" in last_message.additional_kwargs:
                new_memories = self.process_tool_calls(
                    user_id, 
                    contains_information,
                    last_message.additional_kwargs["tool_calls"]
                )
                return new_memories if new_memories else f"无{self.memory_type.value}记忆记录"
            else:
                return f"无{self.memory_type.value}记忆记录"
        return f"无{self.memory_type.value}记忆记录"

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
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tool_input['timestamp'] = current_time
            
            if action == Action.Create:
                self.db_system.add_memory(user_id, category, tool_input)
                print(f"[{current_time}] 成功创建新知识: {tool_input['knowledge']}")
            elif action == Action.Update:
                if 'knowledge_old' in tool_input:
                    self.update_existing_memory(user_id, category, tool_input)
                else:
                    print(f"[{current_time}] 警告: 尝试更新不存在的知识。将其作为新知识添加。")
                    self.db_system.add_memory(user_id, category, tool_input)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 处理知识时出错: {response_dict['message']}")

    def update_existing_memory(self, user_id: str, category: str, memory: Dict[str, Any]):
        memories = self.db_system.get_memories(user_id, category)
        for existing_memory in memories:
            if existing_memory['knowledge'] == memory.get('knowledge_old'):
                new_memory_id = self.db_system.update_memory(
                    user_id, 
                    category, 
                    existing_memory['_id'], 
                    memory
                )
                if new_memory_id:
                    print(f"[{memory.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}] "
                        f"成功更新知识: {memory['knowledge']} (版本ID: {new_memory_id})")
                    return
                
        print(f"警告: 未找到要更新的知识。将其作为新知识添加。")
        self.db_system.add_memory(user_id, category, memory)

def create_memory_system(memory_type: str) -> BaseMemorySystem:
    """Factory function to create the appropriate memory system"""
    try:
        mem_type = MemoryType(memory_type.lower())
        return BaseMemorySystem(mem_type)
    except ValueError:
        raise ValueError(f"Invalid memory type: {memory_type}. Must be either 'explicit' or 'implicit'")

def create_memory_graph(memory_type: MemoryType):
    graph = StateGraph(AgentState)
    # Add nodes
    graph.add_node("sentinel", call_sentinel)
    graph.add_node("knowledge_master", call_knowledge_master)
    graph.add_node("action", call_tool)

    # 设置入口点
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

    # 编译图
    workflow = graph.compile()
    
    # 如果需要使用某些初始状态，可以在调用时传入
    initial_state = {
        "messages": [],
        "memories": {},
        "contains_information": None,
        "memory_type": memory_type
    }
    
    return workflow, initial_state

def get_category_enum(memory_type: MemoryType):
    return ExplicitCategory if memory_type == MemoryType.EXPLICIT else ImplicitCategory

def parse_sentinel_response(response_content: str, memory_type: MemoryType = None) -> Optional[str]:
    response_lower = response_content.lower().strip()
    if "false" in response_lower or "不做操作" in response_lower or "无需新增或修改" in response_lower:
        return None
        
    category_enum = get_category_enum(memory_type) if memory_type else ExplicitCategory
    for category in category_enum:
        if category.value in response_content:
            return category.value
    return None

def call_sentinel(state):
    messages = state["messages"]
    memory_type = state.get("memory_type", MemoryType.EXPLICIT)
    response = state["sentinel_runnable"].invoke(messages)
    category = parse_sentinel_response(response.content, memory_type)
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

if __name__ == "__main__":
    user_id = "test"

    # explicit_system = create_memory_system("explicit")
    # user_input = "我又换了工作地点，我工作地点换到了北京"
    # print("Testing Explicit Memory System:")
    # print(explicit_system.process_user_input(user_id, [user_input]))
    
    implicit_system = create_memory_system("implicit")
    user_input = "我在工作的时候压力很大"
    print("\nTesting Implicit Memory System:")
    print(implicit_system.process_user_input(user_id, [user_input]))

app, initial_state = create_memory_graph(MemoryType.IMPLICIT)