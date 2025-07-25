import math
import aiohttp
import requests
import asyncio
def request(text):
    url = "http://127.0.0.1:8003/embeddings"
    payload = {
        "content" : str(text)#请求的文本（无意义str）
        }
    headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key",#没写key
            "content-type": "application/json"
        }
    response = requests.request("POST", url, json=payload, headers=headers)
    result = response.json()
    return result[0]["embedding"][0]#返回向量
async def main():
    timer = 0#计数器，调试用的
    with open('vector.txt', 'w',encoding='utf-8') as cfile:#清空文件（存向量的）
        cfile.write("")
    with open('map.txt', 'r',encoding='utf-8') as rfile: #打开文件（待转换的）
        line = rfile.readline() #读取文件中的文本，害怕数据量过大一行一行读取
        while line: #读取文件中的文本
            print(f"待转换数据:{line}")#调试用的
            in_embedding = request(line)#转换文本为向量
            with open('vector.txt', 'a') as wfile:#写入文件
                    wfile.write(str(in_embedding)+"\n")#写入向量
            timer+=1#计数器+1
            print("转换完成,转换次数:",timer)#调试用的
            line = rfile.readline()#读取下一行文本
asyncio.run(main())#我也不知道为啥写asyncio
print("转换结束")