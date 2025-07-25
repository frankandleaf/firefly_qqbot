import aiohttp
import ssl
import random
import asyncio
from nonebot import on_regex
from nonebot.typing import T_State
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Bot, Message,Event
from nonebot.adapters.onebot.v11 import MessageSegment
from nonebot.rule import to_me
from typing import Dict
import openai
import logging
import config
#写个日志先
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('firefly.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fireflybot')
#定义ctx字段，如果不加，SSL报错
ctx = ssl.create_default_context()
ctx.set_ciphers('DEFAULT')
client = openai.AsyncOpenAI(
            api_key=config.my_key,
            base_url=config.my_url
            #base_url = config.live_url
        )
reply_msg = on_regex(r".+", priority=10)
group_handlers: Dict[int, GroupMessageHandler] = {}
class GroupMessageHandler:
    def __init__(self,group_id:int):#初始化
        self.response_chance = 99
        self.group_id = group_id
        self.conversation = []
        self.state = 'started'
        logger.info(f'Initialized GroupMessageHandler for group {group_id},state: {self.state}')
    async def handle_message(self, event: GroupMessageEvent, bot: Bot):
        if event.group_id == self.group_id:
            nickname = event.sender.card or event.sender.nickname
            message = event.message.extract_plain_text()
            if message:
                self.conversation.append({"role": "user", "content": nickname+':'+message})
            if random.randint(1,self.response_chance) == 1:
                self.state = 'waiting for response'
                try:
                    response = await client.chat.completions.create(
                        model="firefly",
                        messages=self.conversation,
                        max_tokens=300,
                        temperature=1,
                        frequency_penalty=0.7,
                        presence_penalty=0.3,
                        top_p=0.9,
                        stream=False
                    )
                    reply = response.choices[0].message.content.strip()
                    self.conversation.append({"role": "流萤", "content": reply})
                    await reply_msg.finish(reply) # type: ignore 我说能用就能用
                except Exception as e:
                    print(f"炸了，快去看看为啥：", e)
                    self.state = 'error'
                    bot.send(event,MessageSegment.at(3408662870))
                    return "……（通讯器断开了，也许离心叶有办法？）……"#type: ignore 密码的我说能用就能用
@reply_msg.handle()
async def group_reply_handler(event: GroupMessageEvent, bot: Bot):
    group_id = event.group_id
    handler = group_handlers.get(group_id)
    if not handler:
        handler = GroupMessageHandler(group_id)
        group_handlers[group_id] = handler
    reply = await handler.handle_message(event, bot)
    if reply:
        await reply_msg.finish(reply)
               


