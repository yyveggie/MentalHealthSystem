import asyncio
import websockets
import aioconsole
import json
import sys

async def async_print(*args, **kwargs):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, print, *args, **kwargs)

async def safe_print(*args, **kwargs):
    max_retries = 3
    for _ in range(max_retries):
        try:
            await async_print(*args, **kwargs)
            return
        except BlockingIOError:
            await asyncio.sleep(0.1)
    print("警告：无法打印输出", file=sys.stderr)

async def client():
    uri = "ws://localhost:8765"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                user_id = await aioconsole.ainput("请输入您的用户ID: ")
                consultation_type = int(await aioconsole.ainput("请选择对话类型 (0-7): "))

                await safe_print(f"连接成功！您现在可以开始对话了。输入 '\\exit' 或 '\\结束' 来结束对话。")

                while True:
                    user_input = await aioconsole.ainput(">>: ")

                    if user_input.lower() in ['\\exit', '\\结束']:
                        await safe_print("正在结束对话...")
                        break

                    message = {
                        "user_id": user_id, 
                        "type": consultation_type,
                        "question": user_input
                    }

                    await websocket.send(json.dumps(message))

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30)
                        try:
                            response_data = json.loads(response)
                            await safe_print(response_data)
                            if isinstance(response_data, dict):
                                if 'error' in response_data:
                                    await safe_print(f"\n错误: {response_data['error']}")
                                else:
                                    await safe_print(f"\nEi: {response_data.get('message', response_data)}")
                                    
                                    if response_data.get('tool_data'):
                                        await safe_print(f"\n工具调用: {response_data['tool_data']['tool_name']}")
                                        await safe_print(f"工具输入: {response_data['tool_data']['tool_input']}")
                                        await safe_print(f"工具输出: {response_data['tool_data']['tool_output']}")
                                    
                                    if response_data.get('memory_data'):
                                        await safe_print(f"\n隐式记忆: {response_data['memory_data']['implicit_memory']}")
                                        await safe_print(f"显式记忆: {response_data['memory_data']['explicit_memory']}")
                            else:
                                await safe_print(f"\nEi: {response_data}")
                        except json.JSONDecodeError:
                            await safe_print(f"\nEi: {response}")
                        
                        await safe_print()  # 空行
                    except asyncio.TimeoutError:
                        await safe_print("等待服务器响应超时，正在重新连接...")
                        break

                end_message = {
                    "user_id": user_id,
                    "type": consultation_type,
                    "question": "\\exit"
                }
                await websocket.send(json.dumps(end_message))

                try:
                    final_message = await asyncio.wait_for(websocket.recv(), timeout=5)
                    try:
                        final_data = json.loads(final_message)
                        if isinstance(final_data, dict):
                            await safe_print(f"\nEi: {final_data.get('message', final_data)}")
                        else:
                            await safe_print(f"\nEi: {final_data}")
                    except json.JSONDecodeError:
                        await safe_print(f"\nEi: {final_message}")
                except asyncio.TimeoutError:
                    await safe_print("未收到最后的告别消息")

                break

        except websockets.exceptions.ConnectionClosedError as e:
            await safe_print(f"连接关闭，错误代码：{e.code}，原因：{e.reason}")
            await safe_print("正在尝试重新连接...")
            await asyncio.sleep(5)
        except Exception as e:
            await safe_print(f"发生错误：{str(e)}")
            await safe_print(f"错误类型：{type(e)}")
            import traceback
            await safe_print("详细错误信息：")
            await safe_print(traceback.format_exc())
            await safe_print("正在尝试重新连接...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(client())