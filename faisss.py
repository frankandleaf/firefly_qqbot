import faiss
import numpy as np
import requests
import openai
llm_client = openai.Client(api_key="sk-",base_url="http://127.0.0.1:8002/v1")#初始化openai客户端
index = faiss.IndexFlatL2(1024)# 构建索引(1024维)
def request(text):
    url = "http://127.0.0.1:8003/embeddings"#请求的url，改成你自己的
    payload = {
        "content" : text#请求的文本
        }
    headers = {#请求头
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key",#没写key
            "content-type": "application/json"
        }
    response = requests.request("POST", url, json=payload, headers=headers)#发送请求
    result = response.json()
    return result[0]["embedding"][0]#返回向量

   
with open('/home/frank/fireflybot/fireflybot/plugins/firefly/vector.txt', 'r',encoding='utf-8') as file:#打开文件 
    line = file.readline() #读取文件中的向量
    while line:#读取文件中的向量
        list = []#列表
        data = line.strip("[]\n")#去掉首尾的[]和\n
        list = data.split(", ")#将字符串转换为列表
        #list = [float(i) for i in list] # 将字符串转换为浮点数，他喵的好像不需要
        print(len(list))
        index.add(np.array(list, dtype=np.float32).reshape(1, -1))#添加向量
        line = file.readline()#读取下一个向量
print("索引构建完成")#输出索引构建完成
def check(text,k,max):
    query_vector = np.array(request(text), dtype=np.float32).reshape(1, -1)#输入的文本,转换为向量
    D, I = index.search(query_vector, k)#D是距离，I是索引
    print(D,I)#输出距离和索引,调试用的
    what = []
    with open("/home/frank/fireflybot/fireflybot/plugins/firefly/map.txt", 'r',encoding='utf-8') as file:#打开文件，读取文本，找到最相似的文本
        line = file.readline()#读取文件中的文本，一行一行读取
        counts = 0#计数器
        while line:
            for i in range(len(I[0])):
                if counts == I[0][i] and D[0][i] < max:#如果索引相等且距离小于最大值
                    what.append(line)#将最相似的文本添加到列表中
                    break
            counts += 1
            line = file.readline()
    return what#返回最相似的文本
async def check_with_llm(question,content,num):
    """
    使用LLM检查内容的相似性
    :param content: 输入的文本内容
    :param k: 返回的最相似文本数量
    :param max_distance: 最大距离阈值
    :return: 最相似的文本列表
    """
    content = [c.replace("\n\n", "\n") for c in content]  # 对列表中的每个元素去除换行符
    messages=[
            {"role": "user", "content": f"你是一个文本信息提供者，根据参考内容为名为“流萤”的人回答该问题提供信息，\n【参考内容】：“{content}”问题是：“{question}”"},
        ],
    print(messages)
    completion = llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"你是一个文本信息提供者，根据参考内容为名为“流萤”的人回答该问题提供信息，\n【参考内容】：“{content}”问题是：“{question}”"},
        ],
        temperature=0,
        max_tokens=200,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    response = completion.choices[0].message.content.strip()
    print(f"RAGLLM回复: {response}")  # 调试输出LLM的回复
    num_list = response.split()
    num_list = [int(i) for i in num_list if i.isdigit()]
    return_list = [content[i] for i in num_list if i < len(content)]  # 根据序号获取对应的文本内容
    return return_list  # 返回最相似的文本列表
