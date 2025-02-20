import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import asyncio
import json
import operator
import uuid
from bson import ObjectId
from textwrap import dedent
from datetime import datetime
from typing import Optional, Union, List, Dict, Type, TypedDict, Annotated, Sequence, Tuple

import faiss
import websockets
from colorama import Fore, Style
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, FunctionMessage, AIMessage
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, ToolInvocation, ToolExecutor

from tools import summarize, web_search
from rag.knowledge_graph import retrieve
from memory import explicit_memory, implicit_memory, memory_retrieve
from prompts import guided_conversation, main_system

from load_config import CHAT_MODEL, API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION
from logging_config import setup_logging, disable_logging
import logging
from business.diagnose import MedicalDiagnosisProcessor
from flask import Flask,request

logger = logging.getLogger(__name__)

print("程序开始")
print(API_KEY)
# import warnings
# warnings.filterwarnings("ignore")

local = False

if local:
    # main_llm = ChatOpenAI(temperature=0.7, model=CHAT_MODEL, api_key=API_KEY, base_url=HOST + "/v1")
    # main_llm = ChatOllama(temperature=0.7, model=CHAT_MODEL, base_url=HOST + "/v1")
    main_llm = ChatOpenAI(temperature=0.7, model=CHAT_MODEL, api_key=API_KEY)
else:
    main_llm = ChatOpenAI(temperature=0.7, model="gpt-4o", api_key=API_KEY)

# 上下文记忆设置
# embedding_fn = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=HOST)
embedding_fn = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=API_KEY)
sample_embedding = embedding_fn.embed_query("Sample text")
actual_dimension = len(sample_embedding)

index = faiss.IndexFlatL2(actual_dimension)
vectorstore = FAISS(embedding_function=embedding_fn.embed_query, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})
retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
memory = VectorStoreRetrieverMemory(retriever=retriever)

implicit_memory_knowledge_base = implicit_memory.ImplicitMemorySystem()
explicit_memory_knowledge_base = explicit_memory.ExplicitMemorySystem()


def generate_session_id():
    return str(uuid.uuid4())

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

class Graph_Knowledge_Retrieve(BaseTool):
    name: str = "graph_knowledge_retrieve"
    description: str = "此工具用于检索与特定疾病相关的知识图谱，帮助用户解答关于特定疾病的疑惑。当用户有关于精神疾病的疑问时，调用该工具，如果有返回则结合返回内容和自己的知识回复，如果没有，则使用自己的知识回复"
    class ArgsSchema(BaseModel):
        query: str = Field(..., description="包含特定疾病的实体和关系的查询。例如：抑郁症的治疗方法有哪些?")
    args_schema: Type[BaseModel] = ArgsSchema
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        result = retrieve.run(query)
        return json.dumps({
            "tool_name": self.name,
            "tool_input": query,
            "tool_output": result
        })

class Web_Search(BaseTool):
    name: str = "web_search"
    description: str = f"此工具用于获取最新新闻和信息，帮助用户获取最新信息，你的知识最新到2023年11月，而今天是{datetime.now().strftime('%Y-%m-%d')}。当用户的请求明显要求需要最新的信息支撑时，可以尝试调用该工具。否则，请忽略。"
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
    description: str = """此工具用于从记忆系统中检索用户相关记忆。你需要根据查询内容，从以下类别中选择最相关的类别进行检索：
    情绪体验: 包括当前情绪状态、情绪强度、情绪变化等直接的情感体验
    行为模式: 包括实际的行为反应、应对策略、人际互动方式等可观察的行为
    认知特征: 包括思维方式、信念系统、认知偏差等思维层面的特征
    历史信息: 包括创伤经历、重要生活事件、成长经历等历史性信息
    人格特质: 包括稳定的性格特征、依恋方式、防御机制等
    人口学信息: 包括基本人口统计学特征
    主诉: 包括主要症状和主诉内容
    现病史: 包括当前疾病的发展过程
    用药史: 包括用药情况和药物反应
    物质使用史: 包括成瘾物质的使用情况
    家族史: 包括家庭病史和家庭关系
    社会史: 包括社会功能和社会支持
    创伤史: 包括重大创伤经历
    治疗史: 包括既往治疗经历和效果
    """

    class ArgsSchema(BaseModel):
        categories: List[str] = Field(description="根据查询内容选择的记忆类别列表")
    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, categories: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        memory_system = memory_retrieve.MemoryRetrievalSystem()
        raw_memories = memory_system.retrieve_memories_by_categories(user_id=user_id, categories=categories)
        memories = memory_system.parse_memory_result(raw_memories)
        return json.dumps({
            "tool_name": self.name,
            "tool_input": {
                "user_id": user_id,
                "categories": categories
            },
            "tool_output": memories
        }, ensure_ascii=False)

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
    psy_pred = implicit_memory_knowledge_base.process_user_input(user_id, [user_input])
    print(Fore.BLUE + f"——————————————————————————————————————————————> ||| 隐式记忆推断: {psy_pred}" + Style.RESET_ALL)
    return psy_pred

async def run_memory_read(user_id, user_input):
    exp_pred = explicit_memory_knowledge_base.process_user_input(user_id, [user_input])
    print(Fore.BLUE + f"——————————————————————————————————————————————> ||| 显式记忆推断: {exp_pred}" + Style.RESET_ALL)
    return exp_pred

async def run_handle_conversation(user_input: str, state: AgentState) -> Tuple[AgentState, str, Optional[Dict]]:
    new_state, response, tool_data = await handle_conversation(user_input, state)
    return new_state, response, tool_data

async def websocket_echo(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(message)
        print(f"Sent: {message}")

async def handle_websocket(websocket, path):
        global user_id
        state = None
        print("WebSocket连接已建立，等待用户数据...")
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
                    data = await websocket.recv()
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

                    await websocket.send(json.dumps(response_data, cls=JSONEncoder))

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
    print("Starting WebSocket server on ws://localhost:8763")
    try:
        server = await websockets.serve(handle_websocket, "0.0.0.0", 8763)
        print("WebSocket server started on ws://localhost:8765")
        await server.wait_closed()
    except Exception as e:
        print(f"Error starting WebSocket server: {str(e)}")

async def handle_console_interaction():
    global user_id
    print("\n\n请输入您的用户名或I1D: ")
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

async def handle_console_interactio1():
    print("\n\n请输入您的用户名或I1D:1111 ")

async def main_loop():
    print("程序开始1")
    # 如果你想使用日志（Elasticsearch 或文件）
    _, _ = setup_logging()
    # 或者，如果你想完全禁用日志
    # _, _ = disable_logging()
    logger = logging.getLogger(__name__)
    try:
        print("程序开始2")
        websocket_server = asyncio.create_task(start_websocket_server())
        console_interaction = asyncio.create_task(handle_console_interaction())
        await asyncio.gather(websocket_server,console_interaction)
    except Exception as e:
        print(f"主循环错误: {str(e)}")
        # logger.error(f"主循环错误: {str(e)}")
    finally:
        print("程序结束")
        # logger.info("程序结束")

app = Flask(__name__)
@app.route('/apiv1/diagnosis/processor', strict_slashes=False, methods=['POST'])
def processor_main():
    try:
        fields = request.get_json(force=True)
        token = request.headers.get('X-Ivanka-Token')
        if not token:
            return json.dumps("TOKEN为空",ensure_ascii=False)
        # print("fields")
        # print(fields)
        processor = MedicalDiagnosisProcessor()
        # test_input = {
        #     "过敏史": "药物过敏史：未发现；食物过敏史：否认",
        #     "个人史": "否认长期接触有毒有害物质史，否认严重创伤史，否认长期卧床史，否认手术史。",
        #     "婚育史": "已婚，已育一子",
        #     "家族史": "父母健在，否认家族遗传病史",
        #     "诊疗经过": ""
        # }
        result = processor.process_diagnosis(json.dumps(fields))
        resp = processor.output_format(raw_results=result)
        return json.dumps(resp,ensure_ascii=False)
        #print("\n诊断结果：",resp)
        # if result:
        #     print("\n诊断结果：")
        #     for i, diagnosis in enumerate(result.诊断结果, 1):
        #         print(f"\n可能性 {i}:")
        #         print(f"病症: {diagnosis.病症}")
        #         print(f"置信度: {diagnosis.置信度}")
        #         print(f"理由: {diagnosis.理由}")
        # else:
        #     print("\n未能生成诊断结果")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"\n程序执行出错: {str(e)}")
if __name__ == "__main__":
    # try:
    #     from load_config import WEB_SOCKET_PORT
    #     _, _ = setup_logging()
    #     logger = logging.getLogger(__name__)
    #     server = websockets.serve(handle_websocket, "0.0.0.0", WEB_SOCKET_PORT)
    #     asyncio.get_event_loop().run_until_complete(server)
    #     asyncio.get_event_loop().run_forever()
    # except Exception as e:
    #     print(f"Error starting WebSocket server: {str(e)}")
    #     logger.error(f"Error starting WebSocket server: {str(e)}")
       # asyncio.run(main_loop())
        # asyncio.run(processor_main())
        app.run(debug=False, host='0.0.0.0', port=8763)
