# tools/writing.py
from config import WRITING_KEYS
from models.llm_backend import get_llm_client

def is_writing_intent(text):
    return any(k in text for k in WRITING_KEYS)

def answer_writing(text):
    client, model_name = get_llm_client()
    if client is None:
        return "抱歉，我目前无法进行长篇写作。请配置大模型以启用写作能力。"
    messages = [
        {"role": "system", "content": "你是一个专业的中文写作助手。请根据用户的要求完成写作。文章要有条理、语言流畅、符合指定体裁，不要偷懒，直接输出完整正文。"},
        {"role": "user", "content": text}
    ]
    try:
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.7, max_tokens=1500)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[大模型写作失败] {type(e).__name__}: {e}"