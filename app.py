import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import asyncio
import json
import operator
import uuid
from textwrap import dedent
from datetime import datetime
from typing import Optional, Union, List, Dict, Type, TypedDict, Annotated, Sequence, Tuple

import faiss
import websockets
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, FunctionMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, ToolInvocation, ToolExecutor

from tools import summarize, web_search
from rag.knowledge_graph import retrieve
from memory import explicit_memory, implicit_memory, memory_retrieve
from rag.historical_exp.calculate_similarity import PatientDiagnosisAPI
from prompts import guided_conversation, main_system

from load_config import GPT4O, OPENAI_API_KEY
from logging_config import setup_logging, disable_logging
import logging

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

main_llm = ChatOpenAI(temperature=0.7, model=GPT4O, api_key=OPENAI_API_KEY)

class LocalEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    def __call__(self, text):
        return self.embed_documents([text])[0]
    def embed_documents(self, texts):
        return self.model.encode(texts)
    def embed_query(self, text):
        return self.model.encode([text])[0]

# 上下文记忆设置
embeddings = LocalEmbeddings()
dimension = embeddings.dimension
index = faiss.IndexFlatL2(dimension)
vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})
retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever)

def generate_session_id():
    return str(uuid.uuid4())

class Graph_Knowledge_Retrieve(BaseTool):
    name: str = "graph_knowledge_retrieve"
    description: str = "此工具用于检索与特定疾病相关的知识图谱，帮助用户解答关于特定疾病的疑惑。当用户有关于精神疾病的疑问时，调用该工具，如果有返回则结合返回内容和自己的知识回复，如果没有，则使用自己的知识回复"
    class ArgsSchema(BaseModel):
        query: str = Field(..., description="包含特定疾病的实体和关系的查询。例如：抑郁症的治疗方法有哪些?")
    args_schema: Type[BaseModel] = ArgsSchema
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        result = asyncio.run(retrieve.run(query))
        return json.dumps({
            "tool_name": self.name,
            "tool_input": query,
            "tool_output": result
        })

class Web_Search(BaseTool):
    name: str = "web_search"
    description: str = "此工具用于获取最新新闻和信息，帮助用户获取最新信息。当用户的请求明显要求需要最新的信息支撑时，可以尝试调用该工具。否则，请忽略。"
    class ArgsSchema(BaseModel):
        query: str = Field(..., description="需要在互联网上搜索的完整查询。例如：关于抑郁症的最新新闻有什么？")
    args_schema: Type[BaseModel] = ArgsSchema
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Union[List[Dict], str]:
        result = asyncio.run(web_search.run(query))
        return json.dumps({
            "tool_name": self.name,
            "tool_input": query,
            "tool_output": result
        })

class Memory_Retrieve(BaseTool):
    name: str = "memory_retrieve"
    description: str = "此工具用于从记忆中检索关于用户的记忆，包括个人属性，社交关系，工作状态，心智状态等等。当你不知道用户的一些信息时，调用该工具。"
    class ArgsSchema(BaseModel):
        explicit_memory_query: Optional[str] = Field(None, description="你需要检索的显式记忆。显式记忆是用户的个人属性、家庭属性和社会属性相关的记忆。例如：1.他的年龄是多少？2.他最近的工作是什么？")
        implicit_memory_query: Optional[str] = Field(None, description="你需要检索的隐式记忆。隐式记忆是用户的心理状态、心智能力的历史推论。例如：1. 他最近的心理状态是什么？2. 他具有多重人格吗？")
    args_schema: Type[BaseModel] = ArgsSchema
    def _run(self, explicit_memory_query: Optional[str] = None, implicit_memory_query: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> Union[List[str], str]:
        result = memory_retrieve.run(explicit_memory_query or "", implicit_memory_query or "", user_id)
        return json.dumps({
            "tool_name": self.name,
            "tool_input": {
                "explicit_memory_query": explicit_memory_query,
                "implicit_memory_query": implicit_memory_query
            },
            "tool_output": result
        })

tools = [Graph_Knowledge_Retrieve(), Web_Search(), Memory_Retrieve()]
tool_executor = ToolExecutor(tools=tools)

functions = [convert_to_openai_function(t) for t in tools]
model = main_llm.bind_functions(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    user_id: str
    start_time: datetime

def initialize_state(system_message: SystemMessage, user_id: str) -> AgentState:
    return {
        "messages": [system_message],
        "session_id": generate_session_id(),
        "user_id": user_id,
        "start_time": datetime.now()
    }

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    function_call = last_message.additional_kwargs.get("function_call")
    if not function_call:
        return "end"
    elif function_call["name"] in ["graph_knowledge_retrieve", "web_search", "memory_retrieve"]:
        return "continue"
    else:
        return "end"

def call_model(state):
    messages = state["messages"]
    last_message = messages[-1]
    history = memory.load_memory_variables({"prompt": last_message.content})["history"]
    input_text = f"{messages[0].content}\n{history}\n人类: {last_message.content}\n助手: "
    response = model.invoke(input_text)
    memory.save_context({"input": last_message.content}, {"output": response.content})
    return {"messages": [response]}

def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=response, name=action.tool)
    return {"messages": [function_message]}

async def handle_conversation(user_input: str, state: AgentState) -> Tuple[AgentState, str, Optional[Dict]]:
    response_messages = []
    tool_data = None
    human_message = HumanMessage(content=user_input)
    state["messages"].append(human_message)
    memory.save_context({"input": user_input}, {"output": ""})
    for output in app.stream(state):
        for key, value in output.items():
            if key == "__end__":
                continue
            if isinstance(value, dict) and "messages" in value:
                messages_list = value["messages"]
                for message in messages_list:
                    if isinstance(message, FunctionMessage):
                        try:
                            tool_data = json.loads(message.content)
                        except json.JSONDecodeError:
                            print(f"Warning: Unable to parse FunctionMessage content as JSON: {message.content}")
                            tool_data = {
                                "tool_name": message.name,
                                "tool_output": message.content
                            }
                        
                        ai_input = f"以下是{tool_data['tool_name']}工具返回的结果: </START>{tool_data['tool_output']}</END>\n，请重新组织后继续与用户进行对话，记住，你不需要说明这些信息是来自于哪的，你可以作为自己的知识来运用。"
                        ai_response = await model.ainvoke(ai_input)
                        response_messages.append(ai_response.content)
                    if isinstance(message, AIMessage):
                        response_messages.append(message.content)
    state["messages"] = state["messages"][:1]
    return state, "\n".join(response_messages), tool_data

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", END)
app = workflow.compile()

async def run_psy_predict(user_id, user_input):
    psy_pred = implicit_memory.infer_mental_state(user_id, user_input)
    print(Fore.BLUE + f"——————————————————————————————————————————————> ||| 隐式记忆推断: {psy_pred}" + Style.RESET_ALL)
    return psy_pred

async def run_memory_read(user_id, user_input):
    exp_pred = explicit_memory.record_patient_info(user_id, user_input)
    print(Fore.BLUE + f"——————————————————————————————————————————————> ||| 显式记忆推断: {exp_pred}" + Style.RESET_ALL)
    return exp_pred

async def run_handle_conversation(user_input: str, state: AgentState) -> Tuple[AgentState, str, Optional[Dict]]:
    new_state, response, tool_data = await handle_conversation(user_input, state)
    return new_state, response, tool_data

async def handle_websocket(websocket, path):
    global user_id
    state = None
    
    def choose_consultation_type(type_value):
        if type_value == 0:
            return main_system.main_prompt()
        
        consultation_types = {
            1: guided_conversation.clinical_psychological_consultation,
            2: guided_conversation.marriage_and_family_counseling,
            3: guided_conversation.child_and_adolescent_psychology,
            4: guided_conversation.career_counseling,
            5: guided_conversation.health_psychology,
            6: guided_conversation.addiction_counseling,
            7: guided_conversation.trauma_counseling
        }
        
        return consultation_types.get(type_value, main_system.main_prompt())

    def get_system_prompt(json_data):
        type_value = json_data.get('type', 0)
        return choose_consultation_type(type_value)

    try:
        print("WebSocket连接已建立，等待用户数据...")
        while True:
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=300)  # 5分钟超时
                json_data = json.loads(data)
                print(f"收到数据: {json_data}")
                logger.info(f"接收到的数据 - 用户ID: {json_data.get('user_id')}, 问题: {json_data.get('question')}, 类型: {json_data.get('type')}")
                
                user_id = json_data.get('user_id')
                user_input = json_data.get('question')

                if not user_id or user_input is None:
                    await websocket.send(json.dumps({"error": "无效的数据格式。缺少user_id或question。"}))
                    continue

                if user_input.lower() == "\\exit" or user_input == "\\结束":
                    logger.info(f"对话结束 - 用户ID: {user_id}, 会话ID: {state['session_id'] if state else 'N/A'}")
                    await websocket.send(json.dumps({"message": f"再见👋 {user_id}, 期待我们的下次见面!🥳"}))
                    break

                logger.info(f"用户输入 - 内容: {user_input}, 用户ID: {user_id}, 会话ID: {state['session_id'] if state else 'N/A'}")

                if state is None:
                    system_prompt = get_system_prompt(json_data)
                    system_message = SystemMessage(content=dedent(system_prompt))
                    state = initialize_state(system_message, user_id)
                    logger.info(f"新对话开始 - 用户ID: {user_id}, 会话ID: {state['session_id']}")

                # 检查是否是特殊命令
                if user_input.strip().startswith("请对用户病例信息进行摘要") or user_input.strip().startswith("请你根据住院号为"):
                    response_data = await handle_special_commands(user_input, user_id, state['session_id'], websocket)
                else:
                    psy_pred, exp_pred = await asyncio.gather(
                        run_psy_predict(user_id, user_input),
                        run_memory_read(user_id, user_input)
                    )

                    state, response, tool_data = await run_handle_conversation(user_input, state)
                    
                    response_data = {
                        "message": response,
                        "tool_data": tool_data,
                        "memory_data": {
                            "implicit_memory": psy_pred,
                            "explicit_memory": exp_pred
                        }
                    }

                    await websocket.send(json.dumps(response_data))
                
                logger.info(f"AI响应 - 内容长度: {len(response_data['message'])}, 用户ID: {user_id}, 会话ID: {state['session_id']}")

            except asyncio.TimeoutError:
                logger.warning(f"用户输入超时 - 用户ID: {user_id}, 会话ID: {state['session_id'] if state else 'N/A'}")
                await websocket.send(json.dumps({"message": "您好，您已经很长时间没有发送消息了。如果您还在线，请回复任意消息。"}))

    except websockets.exceptions.ConnectionClosedOK:
        print(f"WebSocket connection closed normally for user: {user_id}")
        logger.info(f"WebSocket连接正常关闭 - 用户ID: {user_id}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error for user: {user_id}. Error: {e}")
        logger.error(f"WebSocket连接异常关闭 - 用户ID: {user_id}, 错误: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in WebSocket communication: {str(e)}")
        logger.error(f"WebSocket通信未预期的错误 - 用户ID: {user_id}, 错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print(f"WebSocket connection closed for user: {user_id}")
        logger.info(f"WebSocket连接已关闭 - 用户ID: {user_id}")

async def start_websocket_server():
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

async def handle_console_interaction():
    global user_id
    print("\n\n请输入您的用户名或ID: ")
    user_id = await asyncio.get_event_loop().run_in_executor(None, input)
    
    guided = await asyncio.get_event_loop().run_in_executor(None, lambda: input("是否需要进行引导性对话测试？（Yes/No）: "))
    guided = guided.lower() == "yes"
    
    if guided:
        system_prompt = guided_conversation.choose_consultation_type()
    else:
        system_prompt = main_system.main_prompt()

    system_message = SystemMessage(content=dedent(system_prompt))
    state = initialize_state(system_message, user_id)
    
    logger.info(f"新对话开始 - 用户ID: {user_id}, 会话ID: {state['session_id']}")

    print("\n--------------------------------------❤️欢迎来到心理治疗室❤️--------------------------------------\n")
    print(f"你好 {user_id}! 我是Ei🙂, 有什么我可以帮助你的吗?\n")

    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, ">>: ")
        if user_input.lower() == "\\exit" or user_input == "\\结束":
            logger.info(f"对话结束 - 用户ID: {user_id}, 会话ID: {state['session_id']}")
            print(f"再见👋 {user_id}, 期待我们的下次见面!🥳")
            break

        logger.info(f"用户输入 - 内容: {user_input}, 用户ID: {user_id}, 会话ID: {state['session_id']}")

        if user_input.strip().startswith("请对用户病例信息进行摘要") or user_input.strip().startswith("请你根据住院号为"):
            response_data = await handle_special_commands(user_input, user_id, state['session_id'])
            print("\nEi: ", response_data['message'])
            if 'tool_data' in response_data:
                print("\n工具调用信息:")
                print(f"工具名称: {response_data['tool_data']['tool_name']}")
                print(f"工具输入: {response_data['tool_data']['tool_input']}")
                print(f"工具输出: {response_data['tool_data']['tool_output']}")
        else:
            psy_pred, exp_pred = await asyncio.gather(
                run_psy_predict(user_id, user_input),
                run_memory_read(user_id, user_input)
            )

            state, response, tool_data = await run_handle_conversation(user_input, state)
            print("\nEi: ", response)
            
            if tool_data:
                print("\n工具调用信息:")
                print(f"工具名称: {tool_data['tool_name']}")
                print(f"工具输入: {tool_data['tool_input']}")
                print(f"工具输出: {tool_data['tool_output']}")
            
            print("\n记忆数据:")
            print(f"隐式记忆: {psy_pred}")
            print(f"显式记忆: {exp_pred}")
            
            logger.info(f"AI响应 - 内容长度: {len(response)}, 用户ID: {user_id}, 会话ID: {state['session_id']}")

        print("——————————————————————————————————————————————>")

async def handle_special_commands(user_input, user_id, session_id, websocket=None):
    response_data = {}
    if user_input.strip().startswith("请对用户病例信息进行摘要"):
        file_path = "./data/hpi.txt"
        tool_name = "summarize"
        tool_input = {"file_path": file_path}
        if os.path.exists(file_path):
            try:
                summarize_prompt = summarize.run(file_path)
                summarize_content = model.invoke(summarize_prompt)
                ai_output = summarize_content.content
                tool_output = summarize_content.content
                logger.info(f"文件总结完成 - 文件: {file_path}, 用户ID: {user_id}, 会话ID: {session_id}")
            except Exception as e:
                error_msg = f"处理文件时出错: {str(e)}"
                tool_output = error_msg
                logger.error(f"文件处理错误 - 文件: {file_path}, 错误: {error_msg}, 用户ID: {user_id}, 会话ID: {session_id}")
        else:
            tool_output = "文件不存在，请检查路径是否正确。"
            logger.warning(f"文件不存在 - 文件: {file_path}, 用户ID: {user_id}, 会话ID: {session_id}")
    
    elif user_input.strip().startswith("请你根据住院号为"):
        json_file_path = "./data/diagnose.json"
        tool_name = "diagnose"
        tool_input = {"file_path": json_file_path}
        historical_exp_api = PatientDiagnosisAPI()
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json_input = json.load(json_file)
                
                vector_results = historical_exp_api.process_query(json.dumps(json_input))
                diagnosis_prompt = main_system.diagnosis_prompt(json_input=json_input, vector_results=vector_results)
                diagnosis = model.invoke(diagnosis_prompt)
                ai_output = diagnosis.content
                tool_output = vector_results
                logger.info(f"诊断完成 - 文件: {json_file_path}, 用户ID: {user_id}, 会话ID: {session_id}")
            except Exception as e:
                error_msg = f"处理JSON文件或进行诊断时出错: {str(e)}"
                tool_output = error_msg
                logger.error(f"诊断错误 - 文件: {json_file_path}, 错误: {error_msg}, 用户ID: {user_id}, 会话ID: {session_id}")
        else:
            tool_output = "文件不存在，请检查路径是否正确。"
            logger.warning(f"诊断文件不存在 - 文件: {json_file_path}, 用户ID: {user_id}, 会话ID: {session_id}")
    
    else:
        tool_name = "unknown_command"
        tool_input = {"command": user_input}
        tool_output = "未知命令"

    response_data = {
        "message": ai_output,
        "tool_data": {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output
        }
    }

    if websocket:
        await websocket.send(json.dumps(response_data))

    return response_data

async def main_loop():
    # 如果你想使用日志（Elasticsearch 或文件）
    # _, _ = setup_logging()
    
    # 或者，如果你想完全禁用日志
    _, _ = disable_logging()

    logger = logging.getLogger(__name__)

    try:
        websocket_server = asyncio.create_task(start_websocket_server())
        console_interaction = asyncio.create_task(handle_console_interaction())
        await asyncio.gather(websocket_server, console_interaction)
    except Exception as e:
        logger.error(f"主循环错误: {str(e)}")
    finally:
        logger.info("程序结束")

if __name__ == "__main__":
    asyncio.run(main_loop())