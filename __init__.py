import aiohttp
import json
import random
import base64  
import asyncio
from nonebot import on_regex
from nonebot.typing import T_State
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Bot, Message,Event
from nonebot.adapters.onebot.v11 import MessageSegment
from nonebot.rule import to_me
from fireflybot.plugins.firefly.faisss import check
import ssl
import openai
ctx = ssl.create_default_context()
ctx.set_ciphers('DEFAULT')
total_token = 0
r_counter = 0
client = openai.AsyncOpenAI(
            api_key="114",
            base_url="http://117.72.123.243:12345/v1"
            #base_url = "http://127.0.0.1:65530/api/oai/v1"
        )
ds_client = openai.AsyncOpenAI(
            api_key="sk-6eb234ad8649472ca29d2baf3d33aef9",
            base_url="https://api.deepseek.com"
)

photo_take = 0

# 全局聊天记录字典
# 群聊：key为group_id，value为聊天记录列表
# 私聊：key为user_id，value为聊天记录列表
import collections
group_conversation_history = collections.defaultdict(list)
private_conversation_history = collections.defaultdict(list)

def clean_current_context(conversation_history):
    """清理聊天记录中的重复'当前情景'，只保留最后一个"""
    context_messages = []
    other_messages = []
    
    for msg in conversation_history:
        if msg.get('role') == 'system' and msg.get('content', '').startswith('当前情景：'):
            context_messages.append(msg)
        else:
            other_messages.append(msg)
    
    # 只保留最后一个当前情景
    if context_messages:
        return [context_messages[-1]] + other_messages
    return other_messages

conversation_history=[]
conversation_init = [
            { 'role': 'system', 'content': '流萤在一个由很多喜欢她的人的凑在一起的群聊中，她喜欢和这些人交流，但她也会因某些不适宜的话题话题感到不适而试图阻止群内的讨论\n【流萤的记忆】代表流萤知道的信息，她会参考这些信息来回答问题\n【流萤的记忆】代表流萤脑中浮现的经历和她对世界的看法，她会根据这些信息聊天（信息和问题无关时不强用）\n流萤喜欢群里的大家，感谢大家的支持'},
        ]
conversation_user_init = [
            { 'role': 'system', 'content': '流萤在一个私聊中，她会因某些不适宜的话题话题感到不适而试图阻止当前话题的讨论或直接表达对对方冒犯性语言的不满。\n【流萤的记忆】的内容代表流萤脑中浮现的经历和她对世界的看法，她会参考这些信息聊天（信息和问题无关时忽略）\n流萤和对方相互喜欢'},
        ]
file_path = r'./chat_log.txt'

getmsg = on_regex(pattern=r'.*', priority=0)
@getmsg.handle()
async def handle_image(bot: Bot, event: Event):
    clear_signal = 0
    reply_judge = 0 #回复概率 
    nickname = event.sender.card or event.sender.nickname
    # 判断消息类型
    is_group = hasattr(event, 'group_id')
    if is_group:
       
        key = event.group_id
        conversation_history = group_conversation_history[key]
        if not conversation_history:
            conversation_history.extend(conversation_init)
        # 判断是否被@
        if hasattr(event, 'to_me') and event.to_me:
            # 被@，必然回复
            pass
        else:
            # 未被@，按概率回复
            if random.random() > reply_judge:
                conversation_history.append({
                    "role": "user",
                    "content": f"{event.sender.card or event.sender.nickname}: {event.message.extract_plain_text()}"
                })
                print(conversation_history)#debug
                print("未被@，不回复")
                return
    else:
        key = event.user_id
        conversation_history = private_conversation_history[key]
        if not conversation_history:
            conversation_history.extend(conversation_user_init)
    if len(conversation_history) >= 35:
        conversation_history_backup = conversation_history[-10:].copy()
        ds_context_response = await ds_context(nickname, conversation_history[1:35])
        
        # 清空并重新初始化
        conversation_history.clear()
        conversation_history.extend(conversation_user_init if not is_group else conversation_init)
        
        if ds_context_response is not False:
            conversation_history.append({'role': 'system', 'content': "当前情景：" + (ds_context_response or "")})
        
        conversation_history.extend(conversation_history_backup)
        conversation_history[:] = clean_current_context(conversation_history)
        print("ds_context_response:", ds_context_response)  # debug
        print("conversation_history:",conversation_history)  # debug

    message = event.message
    images = [
        seg.data["url"]
        for seg in message
        if seg.type == "image"
    ]
    user_message = event.message.extract_plain_text()
    messages_to_send = []
    if 0:#turn off this function(图片处理功能)
        for url in images:
            try:
                urll = str(url.replace("\\",""))
                urll = urll.replace("\\","")
                save_path = "/home/frank/temp_image.png"
                await download_image(urll, save_path)
                with open(save_path, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                conversation_history.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{nickname}: {user_message}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
                })
            except Exception as e:
                print("图片处理过程出现问题：", str(e))
    else:
        print("消息中没有包含图片。")

    response = await get_model_response(user_message, nickname, conversation_history,is_group)
    for i in split_message(response):
        reply_seg = MessageSegment.reply(event.message_id)
        await bot.send(event=event, message=Message(Message(i)))
        await asyncio.sleep(1)  # 确保不会发送过快
    try:
        await gptsovits(response)
        wav_path = '/home/frank/temp_voice/temp_out.wav'
        await bot.send(event, message=MessageSegment.record(wav_path))
    except Exception as e:
        print(f"语音生成或发送失败: {str(e)}")

async def download_image(url: str, save_path: str):#url:图片链接，save_path:保存路径
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=ctx) as response:  # 修正参数名为ssl
            if response.status == 200:
                #写入二进制内容到指定文件
                with open(save_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"HTTP 错误，状态码：{response.status}")


async def ds_context(nickname = "离心叶", conversation_history=None):
    if conversation_history is None or len(conversation_history) < 2:
        return False
    q = str(conversation_history[1:])+f"\n\n用稍简洁的语言，以流萤的角度概括聊天记录（以\"流萤\"替换\"我\"，保留关键主体，去除细枝末节，回答应为纯文本格式，只包含一段文字，不超过200字，作为模型清空聊天记录的下一轮的prompt"
    messages = [
        {"role": "user", "content": q}
    ]
    print(messages)
    completion = await ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=300,
        frequency_penalty=0,
        presence_penalty=0,
        temperature=0.4,
        stream=False,
    )
    response = completion.choices[0].message.content
    return response
async def get_model_response(user_message: str, nickname = "离心叶", conversation_history=None, is_group=False):
    global r_counter
    global photo_take
    if conversation_history is None:
        conversation_history = []
    
    # 初始化 RAG_list
    RAG_list = None
    
    # Token management
    # Only append text message if no image was sent
    if not any(isinstance(msg['content'], list) for msg in conversation_history[-1:]):
        RAG_Q = user_message.replace("你","流萤").replace("我","离心叶").replace(nickname+":","")
        RAG_list = check(RAG_Q, 2, 0.9)
        if RAG_list is not None and len(RAG_list) > 0:
            pass
        if len(RAG_list) > 0:
            RAG_str = "\n".join(RAG_list)
            conversation_history.append({
                "role": "user",
                "content": f"【流萤的记忆】{RAG_str}\n{nickname}: {user_message}"
            })
        else:
            conversation_history.append({
                "role": "user",
                "content": f"{nickname}: {user_message}"
            })
    else:
        conversation_history.append({
            "role": "user",
            "content": f"{nickname}: {user_message}"
        })
    completion = await client.chat.completions.create(
        model="output",
        messages=conversation_history,
        max_tokens=300,
        frequency_penalty=1.0,
        presence_penalty=0.3,
        temperature=0.7,
        top_p=0.7,
        stream=False,
    )
    response = completion.choices[0].message.content
    print(f"模型回复: {response}")
    if RAG_list is not None and len(RAG_list) > 0:
        conversation_history.pop(-1)  # 删除第一条消息
        conversation_history.append({
            "role": "user",
            "content": f"{nickname}: {user_message}"
        })
    if len(response) <20:
        r_counter +=1
    else:
        r_counter = 0
    if r_counter > 4:
        print("模型回复过短，可能需要清空聊天记录")
        ds_context_response = await ds_context(nickname, conversation_history)
        conversation_history.clear()
        # 根据会话类型选择正确的初始化
        conversation_history.extend(conversation_user_init if not is_group else conversation_init)
        if ds_context_response is not False:
            conversation_history.append({
                "role": "system",
                "content": "当前情景：" + (ds_context_response or "")
            })
        # 清理可能的重复当前情景
        conversation_history[:] = clean_current_context(conversation_history)
    
    if response is not None:
        response = response.replace("<think>","").replace("</think>","").replace("\n\n","").replace("流萤:","")
    else:
        response = ""
    conversation_history.append({"role": "model", "content": response})
    return response


async def embeddings(text):
    API_URL="http://192.168.1.42:8003/embedding"
    API_KEY=""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    payload = {
        'content':text
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload, ssl=False) as response:
            print(response.status)
            print(response.text)
            result =await response.json()
            embedding = result[0]["embedding"]
            return embedding  




#用语音合成功能的函数——————————————————————————————————————————————————————————————————————————————————————————
async def gptsovits(gottext: str) -> str:  # 语音合成，你不用管，反正咋地时间都那样
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
        async with session.post("http://127.0.0.1:9880/tts", json=payload) as response:  #发个aio的http请求
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

def split_message(message: str) -> list:
    # 定义分割点标记
    split_marks = ['。','!','.', '！', '？', '...']
    # 找出所有可能的分割点
    split_positions = []
    for i, char in enumerate(message):
        if char in split_marks or (char == '.' and i + 2 < len(message) and message[i:i+3] == '...'):
            if char == '.' and i + 2 < len(message) and message[i:i+3] == '...':
                split_positions.append(i+2)
            else:
                split_positions.append(i)
    
    if not split_positions:
        return [message]
        
    # 随机决定分成几段（1-5段）
    num_segments = min(random.randint(1, 5), len(split_positions) + 1)
    
    # 随机选择分割点
    if num_segments > 1:
        selected_positions = sorted(random.sample(split_positions, num_segments - 1))
        
        # 分割消息
        segments = []
        start = 0
        for pos in selected_positions:
            segments.append(message[start:pos+1].strip())
            start = pos+1
        segments.append(message[start:].strip())
        return [seg for seg in segments if seg]
    
    return [message]
