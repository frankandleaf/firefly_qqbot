import base64
import requests
import os
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
import fireflybot.plugins.firefly.llm_config as llm_config

model_url = llm_config.model_url+"/chat/completions"
model_key = llm_config.model_key
model_name = llm_config.model_name


def image_process(image_path: str, conversation_history: List[Dict[str, Any]], user_message: str, nickname: str) -> Dict[str, Any]:
    """
    处理图片并返回模型回复
    """
    try:
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # 构建包含图片的消息
        message_content = [
            {"type": "text", "text": f"{nickname}: {user_message}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
        
        # 添加用户消息到对话历史
        conversation_history.append({
            "role": "user",
            "content": message_content
        })
        
        url = model_url
        headers = {
            "Authorization": "Bearer " + model_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": conversation_history,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"图片处理API调用失败: {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        print(f"图片处理过程中出现错误: {str(e)}")
        return None


async def image_process_async(image_path: str, conversation_history: List[Dict[str, Any]], user_message: str, nickname: str) -> Dict[str, Any]:
    """
    异步版本的图片处理函数
    """
    try:
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # 构建包含图片的消息
        message_content = [
            {"type": "text", "text": f"{nickname}: {user_message}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
        ]
        
        # 添加用户消息到对话历史
        conversation_history.append({
            "role": "user",
            "content": message_content
        })
        
        url = model_url
        headers = {
            "Authorization": "Bearer " + model_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": conversation_history,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=30) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"图片处理API调用失败: {response.status}, {await response.text()}")
                    return None
                    
    except Exception as e:
        print(f"图片处理过程中出现错误: {str(e)}")
        return None