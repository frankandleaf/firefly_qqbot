context_key = "your_context_key"#上下文整合模型的key
context_url = "https://your_context_url"#上下文整合模型的url
model_key = "sk-1145141919810"#模型本体的key
model_url = "http://your_model_url"#模型本体的url
group_system_prompt = [{'role':'system','content':'口瓜'}]#模型本体对于用户的提示词(列表，格式就这样就行，也可以写多个)
user_system_prompt = [{'role':'system','content':'口瓜'}]#模型本体对于私聊提示词(列表，格式就这样就行，也可以写多个)
reply_chance = 0.1#对于群聊消息的回复概率
context_model = "deepseek-chat"#上下文整合模型的模型名
context_prompt = ""#上下文整合模型的提示词
context_max_tokens = 300#上下文整合模型的最大token数
context_temperature = 0.4#上下文整合模型的温度
context_frequency_penalty = 0#上下文整合模型的频率惩罚
context_presence_penalty = 0#上下文整合模型的存在惩罚
model_max_tokens = 300#模型本体的最大token数
model_temperature = 0.7#模型本体的温度
model_frequency_penalty = 1.0#模型本体的频率惩罚
model_presence_penalty = 0.3#模型本体的存在惩罚
memory_prifix = "【流萤的记忆】"#记忆的提示词前缀
model_name = "666"#模型本体的名称
wav_path = "/home/frank/temp_voice/temp_out.wav"#聊天中语音合成后的文件路径
save_path = "/home/frank/temp_image.png"  # 聊天中图片保存路径










