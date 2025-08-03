import base64
import requests
import fireflybot.plugins.firefly.llm_config as llm_config
# 读取图片并转为 base64
save_path = "example.jpg"
with open(save_path, 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# 构造请求
url = llm_config.model_url
headers = {
    "Authorization": "Bearer 1234567890",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen2.5vl:7b",
    "messages": [
        {"role":"system","content":"图片中央的女孩是流萤自己"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "离心leaf:你是谁"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }
    ]
}

# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)
print(response.json())