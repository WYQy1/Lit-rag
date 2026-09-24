# prompt_lab.py — Prompt 基础实验台（W2 · 学 CoT 用）
# 直接运行，看 3 组对比实验的结果
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def ask(system, user):
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
    )
    return resp.choices[0].message.content

SEP = "=" * 40

# ===== 实验 1：要不要加"一步步思考"（零样本 CoT）=====
print(SEP + "\n实验1a：直接问（不加任何提示）\n" + SEP)
q = "一个房间里有3盏灯和3个开关，每个开关控制一盏灯。你只能进房间一次，怎么确定每个开关对应哪盏灯？"
print(ask("你是一个乐于助人的助手。", q))

print("\n" + SEP + "\n实验1b：问题末尾加'请一步一步思考，再给出答案'\n" + SEP)
print(ask("你是一个乐于助人的助手。", q + "\n请一步一步思考，再给出答案。"))

# ===== 实验 2：给一个带推理过程的例子（少样本 CoT）=====
print("\n" + SEP + "\n实验2：先给一个带推理步骤的例子，再问新题\n" + SEP)
print(ask(
    "你是一个乐于助人的助手。",
    "例子：\n"
    "问题：小明有5个苹果，吃了2个，又买了3个，现在几个？\n"
    "推理：5-2=3，3+3=6。\n"
    "答案：6。\n\n"
    "现在请回答：一根绳子对折3次后，从中间剪断，一共分成几段？\n"
    "请先给出推理过程，再给出答案。"
))

# ===== 实验 3：角色设定 + 结构化要求（贴近你课题）=====
print("\n" + SEP + "\n实验3：给模型设定'审稿人'角色，并要求按结构回答\n" + SEP)
print(ask(
    "你是一位光学工程领域的资深审稿人，说话专业、客观。请严格按以下结构输出：\n"
    "1. 创新性评价（1-2句）\n"
    "2. 可行性问题（1-2句）\n"
    "3. 改进建议（1-2句）",
    "请评价这个研究思路：用 CoT 的方式分析：Cr3+ 掺杂浓度对近红外发射峰位和强度的影响规律有哪些？"
))
