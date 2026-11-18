# router.py
from config import PRONOUNS, KNOWLEDGE_HINTS, REFUSE_TEXT
from tools.time_calc import is_time_intent, answer_time, is_calc_intent, answer_calc
from tools.weather import is_weather_intent, is_weather_followup, answer_weather
from tools.writing import is_writing_intent, answer_writing
from knowledge_base.rag import answer_rag
from models.llm_backend import get_llm_client, llm_chat
from models.local_seq2seq import local_chat_with_conf

class ChatSession:
    def __init__(self, max_turns=6):
        self.history = []
        self.max_turns = max_turns
    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]
    def recent_user_texts(self, n=2):
        return [m["content"] for m in self.history if m["role"] == "user"][-n:]
    def as_messages(self):
        return list(self.history)

def build_rag_query(text, session=None):
    # 1. 极短否定词直接返回
    negative_words = ["no", "不是", "不对", "错", "没有", "不"]
    if text.lower() in negative_words or len(text) <= 2:
        return text
        
    if session is None or not session.history: 
        return text
        
    # 2. 如果句子超过了 6 个字，说明用户说了一个完整的新问题，不需要强行拼接上下文
    if len(text) > 6:
        return text
        
    # 3. 只有非常短的问题（比如“那个呢”、“多少钱”），且包含代词时，才借用上一轮
    if any(p in text for p in PRONOUNS):
        prev = session.recent_user_texts(1)
        if prev: 
            return prev[0] + " " + text
            
    return text

def is_knowledge_like(text):
    return any(k in text for k in KNOWLEDGE_HINTS)

def chat_response(text, local_model=None, use_llm=True, session=None):
    if not text or not text.strip(): return "你好像什么都没说。"
    
    # 1. 本地工具
    if is_time_intent(text): return answer_time(text)
    if is_calc_intent(text): return answer_calc(text)
    if is_weather_intent(text) or is_weather_followup(text, session): return answer_weather(text, session=session)
    if is_writing_intent(text): return answer_writing(text)
    
    # 2. 【新增】闲聊与寒暄拦截
    # 纯打招呼、道谢、告别等，直接跳过 RAG，防止误触发知识库
    GREETINGS = ["你好", "您好", "嗨", "在吗", "hello", "hi", "你是谁", "谢谢", "再见", "晚安", "早上好"]
    is_chitchat = any(g == text.strip().lower() for g in GREETINGS) or len(text.strip()) <= 2
    
    if not is_chitchat:
        # 只有不是闲聊，才去查知识库
        rag_query = build_rag_query(text, session)
        status, rag_ans, _ = answer_rag(rag_query, original_text=text, use_llm=use_llm, history=session.as_messages() if session else None)
        if status in ("hit", "weak"): 
            return rag_ans
        if is_knowledge_like(text): 
            return REFUSE_TEXT
    
    # 3. 大模型兜底
    client, _ = get_llm_client()
    if use_llm and client is not None:
        ans = llm_chat(text, history=session.as_messages() if session else None)
        if ans: return ans
        
    # 4. 本地模型兜底
    if local_model is not None: 
        return local_chat_with_conf(text, local_model)[0]
        
    return "我暂时无法回答。"