import re
import aiohttp
import json
import random
import base64  
import asyncio
from nonebot import on_message
from nonebot.typing import T_State
from nonebot.adapters.onebot.v11 import Bot, Message,MessageEvent
from nonebot.adapters.onebot.v11 import MessageSegment
from nonebot.rule import to_me
from fireflybot.plugins.firefly.faisss import check
import ssl
import openai
# 导入图片处理模块
import fireflybot.plugins.firefly.image_process as image_process

# ================== 配置与全局变量 ==================
import fireflybot.plugins.firefly.llm_config as llm_config

ssl_ctx = ssl.create_default_context()
ssl_ctx.set_ciphers('DEFAULT')

total_token = 0
r_counter = 0
user_tasks = {}  # user_id: asyncio.Task

client = openai.AsyncOpenAI(
    api_key=llm_config.model_key,
    base_url=llm_config.model_url
)
ds_client = openai.AsyncOpenAI(
    api_key=llm_config.context_key,
    base_url=llm_config.context_url
)

photo_take = 0

# 聊天历史
import collections
group_conversation_history = collections.defaultdict(list)
private_conversation_history = collections.defaultdict(list)

conversation_init = llm_config.group_system_prompt
conversation_user_init = llm_config.user_system_prompt
file_path = r'./chat_log.txt'

# ================== 工具函数 ==================

def clean_current_context(conversation_history):
    """清理聊天记录中的重复'当前情景'，只保留最后一个"""
    context_messages = []
    other_messages = []
    for msg in conversation_history:
        if msg.get('role') == 'system' and msg.get('content', '').startswith('当前情景：'):
            context_messages.append(msg)
        else:
            other_messages.append(msg)
    if context_messages:
        return [context_messages[-1]] + other_messages
    return other_messages

async def download_image(url: str, save_path: str):
    """下载图片到本地"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=ssl_ctx) as response:
            if response.status == 200:
                with open(save_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"HTTP 错误，状态码：{response.status}")

def split_message(message: str) -> list:
    """按中英文标点分割消息"""
    sentences = re.split(r'(?<=[。！？!?~])', message)
    return [s.strip() for s in sentences if s.strip()]

# ================== 语音合成 ==================

async def gptsovits(gottext: str) -> str:
    """调用GPT-SoVits语音合成"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "text": gottext,
            "text_lang": "zh",
            "ref_audio_path": "/home/frank/GPTSoVits/example.wav",
            "aux_ref_audio_paths": [],
            "prompt_text": "我想，沐浴在光中的人，一定会被那种温暖所吸引吧",
            "prompt_lang": "zh",
            "top_k": 7,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut5",
            "batch_size": 20,
            "speed_factor": 1,
            "ref_text_free": False,
            "split_bucket": True,
            "fragment_interval": 0.3,
            "seed": -1,
            "keep_random": True,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35
        }
        async with session.post("http://127.0.0.1:9880/tts", json=payload) as response:
            if response.status == 200:
                content_type = response.headers.get('Content-Type')
                if content_type and 'audio/wav' in content_type:
                    audio_data = await response.read()
                    with open("/home/frank/temp_voice/temp_out.wav", "wb") as f:
                        f.write(audio_data)
                    print("Audio saved")
                    return "Audio saved as output.wav"
                else:
                    result = await response.json()
                    audio_url = result.get('audio_url', '')
                    return audio_url
            else:
                raise Exception(f"语音合成失败: {response.status}, {await response.text()}")

# ================== 上下文摘要 ==================

async def ds_context(nickname="", conversation_history=None):
    """调用摘要模型，生成当前情景"""
    if conversation_history is None or len(conversation_history) < 2:
        return False
    q = str(conversation_history[1:]) + llm_config.context_prompt
    messages = [{"role": "user", "content": q}]
    print(messages)
    completion = await ds_client.chat.completions.create(
        model=llm_config.context_model,
        messages=messages,
        max_tokens=llm_config.context_max_tokens,
        frequency_penalty=llm_config.context_frequency_penalty,
        presence_penalty=llm_config.context_presence_penalty,
        temperature=llm_config.context_temperature,
        stream=False,
    )
    response = completion.choices[0].message.content
    return response

# ================== 主消息处理 ==================
message_matcher = on_message()
@message_matcher.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    """主入口，处理消息事件"""
    user_id = event.get_user_id()
    old_task = user_tasks.get(user_id)
    if old_task and not old_task.done():
        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            print("取消")
    loop = asyncio.get_running_loop()
    task = loop.create_task(process_user_request(bot, event))
    user_tasks[user_id] = task

def append_message(conversation_history, role, content, image=""):
    """保证user/assistant交替添加消息"""
    try:
        if not conversation_history or conversation_history[-1].get("role") != role:
            if image:
                # 把图片base64以字符串形式拼接到content
                conversation_history.append({"role":"user","content":[{"type":"text","text":content},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image}"}}]})
            else:
                conversation_history.append({
                    "role": role,
                    "content": content
                })
        else:
            last_msg = conversation_history[-1]["content"]
            conversation_history.pop()
            # 如果最后一条消息包含图片，不能简单合并文本
            if isinstance(last_msg, list):
                # 如果最后一条是图片消息，创建新的消息
                conversation_history.append({
                    "role": role,
                    "content": content
                })
            else:
                # 合并文本
                conversation_history.append({"role": role, "content": f"{last_msg}\n{content}"})
    except Exception as e:
        print(f"添加消息出现问题:{e}")

async def process_user_request(bot: Bot, event: MessageEvent):
    """处理用户请求，包含消息分发、上下文管理、回复、语音合成"""
    reply_judge = 0  # 回复概率
    nickname = event.sender.card or event.sender.nickname
    is_group = hasattr(event, 'group_id')
    key = event.group_id if is_group else event.user_id
    conversation_history = group_conversation_history[key] if is_group else private_conversation_history[key]
    if not conversation_history:
        conversation_history.extend(conversation_init if is_group else conversation_user_init)
    if is_group:
        if random.randint(1,100) > reply_judge:
            return
        else:
            append_message(conversation_history,"user",event.message.extract_plain_text())

    # 聊天历史过长，摘要
    if len(conversation_history) >= 35:
        conversation_history_backup = conversation_history[-10:].copy()
        ds_context_response = await ds_context(nickname, conversation_history[1:35])
        conversation_history.clear()
        conversation_history.extend(conversation_user_init if not is_group else conversation_init)
        if ds_context_response is not False:
            #conversation_history.append({'role': 'system', 'content': "当前情景：" + (ds_context_response or "")})
            append_message(conversation_history,"system",f"当前情景:{ds_context_response or ''}")
        conversation_history.extend(conversation_history_backup)
        conversation_history[:] = clean_current_context(conversation_history)
        print("ds_context_response:", ds_context_response)
        print("conversation_history:", conversation_history)

    # 处理图片
    message = event.message
    images = [seg.data["url"] for seg in message if seg.type == "image"]
    user_message = event.message.extract_plain_text()
    img_base64 = ""
    has_image = False
    save_path = llm_config.save_path  # 初始化save_path
    
    if images:
        has_image = True
        for url in images:
            try:
                urll = str(url.replace("\\", ""))
                await download_image(urll, save_path)
                # 使用image_process模块处理图片
                response_data = await image_process.image_process_async(save_path, conversation_history, user_message, nickname)
                if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
                    response = response_data['choices'][0]['message']['content']
                    if response and isinstance(response, str):  # 确保response不为None且为字符串
                        print(f"图片处理回复: {response}")
                        # 直接发送回复，不需要再次调用模型
                        for i in split_message(response):
                            await asyncio.sleep(len(i) /2 * 0.8)  # 拟人化的等待
                            await bot.send(event=event, message=Message(i))
                            append_message(conversation_history,"assistant",i)
                    
                    # 语音合成与发送
                    try:
                        await gptsovits(response)
                        wav_path = llm_config.wav_path
                        await bot.send(event, message=MessageSegment.record(wav_path))
                    except Exception as e:
                        print(f"语音生成或发送失败: {str(e)}")
                    return  # 图片处理完成，直接返回
                else:
                    print("图片处理失败，使用默认处理方式")
                    with open(save_path, 'rb') as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    append_message(conversation_history,"user",f"{nickname}: {user_message}",img_base64)
            except Exception as e:
                print("图片处理过程出现问题：", str(e))
                # 如果图片处理失败，使用默认方式
                try:
                    with open(save_path, 'rb') as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    append_message(conversation_history,"user",f"{nickname}: {user_message}",img_base64)
                except:
                    pass
    
    # 如果没有图片或图片处理失败，使用默认处理方式
    if not has_image:
        append_message(conversation_history,"user",f"{nickname}: {user_message}")
    
    # 获取模型回复
    response = await get_model_response(user_message, nickname, conversation_history, is_group,img_base64)
    img_base64 = ""
    for i in split_message(response):
        await asyncio.sleep(len(i) /2 * 0.8)  # 拟人化的等待
        await bot.send(event=event, message=Message(i))
        append_message(conversation_history,"assistant",i)

    # 语音合成与发送
    try:
        await gptsovits(response)
        wav_path = '/home/frank/temp_voice/temp_out.wav'
        await bot.send(event, message=MessageSegment.record(wav_path))
    except Exception as e:
        print(f"语音生成或发送失败: {str(e)}")

async def get_model_response(user_message: str, nickname="离心叶", conversation_history=None, is_group=False,img=""):
    """调用主模型，获取回复"""
    global r_counter
    if conversation_history is None:
        conversation_history = []
    RAG_list = None
    
    # 检查最后一条消息是否包含图片
    has_image_in_last_message = False
    if conversation_history and isinstance(conversation_history[-1].get('content'), list):
        has_image_in_last_message = any(item.get('type') == 'image_url' for item in conversation_history[-1].get('content', []))
    
    # 只在最后一条不是图片时做RAG
    if not has_image_in_last_message:
        RAG_list = process_rag(user_message, nickname, conversation_history,img)
    
    completion = await client.chat.completions.create(
        model="output",
        messages=conversation_history,
        max_tokens=llm_config.model_max_tokens,
        frequency_penalty=llm_config.model_frequency_penalty,
        presence_penalty=llm_config.model_presence_penalty,
        temperature=llm_config.model_temperature,
        top_p=0.9,
        stream=False,
    )
    response = completion.choices[0].message.content
    print(f"模型回复: {response}")
    
    if RAG_list and not has_image_in_last_message:
        conversation_history.pop()
        append_message(conversation_history,"user",f"{nickname}: {user_message}")
    
    if response and len(response) < 20:
        r_counter += 1
    else:
        r_counter = 0
    if r_counter > 4:
        print("模型回复过短，可能需要清空聊天记录")
        ds_context_response = await ds_context(nickname, conversation_history)
        conversation_history.clear()
        conversation_history.extend(conversation_user_init if not is_group else conversation_init)
        if ds_context_response is not False:
            append_message(conversation_history,"system",f"当前情景:{ds_context_response or ''}")
        conversation_history[:] = clean_current_context(conversation_history)
    else:
        response = ""
    return response

############RAG#################
def process_rag(user_message, nickname, conversation_history,img = ""):
    """
    封装RAG处理逻辑，返回RAG_list，并根据RAG结果更新conversation_history
    """
    RAG_list = None
    # 只在最后一条不是图片时做RAG
    if not any(isinstance(msg['content'], list) for msg in conversation_history[-1:]):
        RAG_Q = user_message.replace(nickname + ":", "") 
        RAG_list = check(RAG_Q, 3, 0.9)
        if RAG_list:
            RAG_str = "\n".join(RAG_list)
            append_message(conversation_history, "user", f"{nickname}:{user_message}\n{llm_config.memory_prifix}:{RAG_str}【记忆结束】",img)
        else:
            append_message(conversation_history, "user", f"{nickname}: {user_message}",img)
    else:
        append_message(conversation_history, "user", f"{nickname}: {user_message}",img)
    return RAG_list
