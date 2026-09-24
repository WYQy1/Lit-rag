# hello_llm.py — 我的第一个大模型 API 调用（DeepSeek）
# 模型：deepseek-flash（便宜款，2026.9 官方在售）
# 前提：同目录下有 .env 文件，内容为 DEEPSEEK_API_KEY=sk-你的key

import os

from dotenv import load_dotenv
from openai import OpenAI

# 1. 从 .env 读取 API Key（不要把 key 直接写进代码）
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise SystemExit("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件是否写对")

# 2. 创建客户端：base_url 指向 DeepSeek 官方接口
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 3. 发送一次对话请求
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    ],
    stream=False,
)

# 4. 打印模型的回复
print("模型回复：", response.choices[0].message.content)
print("本次用量（tokens）：", response.usage)
