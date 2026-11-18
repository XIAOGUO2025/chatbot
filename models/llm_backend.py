# models/llm_backend.py
import os
from config import LLM_BACKEND

_LLM_CLIENT = None
_LLM_MODEL = None

def init_llm():
    global _LLM_CLIENT, _LLM_MODEL
    if LLM_BACKEND is None: return
    try:
        from openai import OpenAI
    except ImportError:
        print("[警告] 未安装 openai，大模型不可用。pip install openai")
        return
    if LLM_BACKEND == "deepseek":
        _LLM_CLIENT = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        _LLM_MODEL = "deepseek-chat"
    elif LLM_BACKEND == "openai":
        _LLM_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _LLM_MODEL = "gpt-4o-mini"

def get_llm_client():
    return _LLM_CLIENT, _LLM_MODEL

def llm_chat(prompt, history=None):
    if _LLM_CLIENT is None: return None
    messages = [{"role": "system", "content": "你是友好、简洁的中文助手。"}]
    if history: messages.extend(history[-6:])
    messages.append({"role": "user", "content": prompt})
    try:
        resp = _LLM_CLIENT.chat.completions.create(model=_LLM_MODEL, messages=messages, temperature=0.3, max_tokens=400)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[大模型失败] {e}"