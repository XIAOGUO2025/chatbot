"""
完整对话助手（带防乱说机制）：
  - 天气工具（Open-Meteo API，支持多轮城市指代）
  - 知识库 RAG（纯文本 knowledge.txt，字符 bigram TF-IDF）
  - 三态检索：hit / weak / miss，没证据不硬答
  - 知识类问题拒答：知识库没有可靠信息时明确说不知道
  - 引用校验（可选）：大模型回答后再核对是否被资料支持
  - 本地 Seq2Seq（带置信度拒答，单轮兜底）
  - 会话上下文 ChatSession

运行:
    pip install torch requests
    pip install openai          # 可选，用于大模型后端和引用校验
    python chatbot_rag.py

环境变量（可选）:
    DEEPSEEK_API_KEY=...
    OPENAI_API_KEY=...
"""
import math
import os
import random
import re
import time
from collections import Counter

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
random.seed(42)


# ============================================================
# 0. 配置区
# ============================================================
DEFAULT_CITY = "北京"
KNOWLEDGE_FILE = "knowledge.txt"

RAG_TOP_K = 3
RAG_SCORE_MISS = 0.10      # 低于此视为完全无关
RAG_SCORE_HIT = 0.25       # 高于此视为可靠命中；中间为 weak

ENABLE_VERIFY = False      # True 时对 RAG 回答做引用校验（多一次大模型调用）
LOCAL_CONF_THRESHOLD = 0.5  # 本地模型平均置信度低于此值则拒答

LLM_BACKEND = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else (
    "openai" if os.environ.get("OPENAI_API_KEY") else None
)

WEATHER_KEYS = ["天气", "气温", "温度", "下雨", "下雪", "刮风",
                "冷不冷", "热不热", "穿什么"]

PRONOUNS = ["它", "他", "她", "这个", "那个", "这", "那",
            "再", "还", "另外", "上面", "前面", "呢"]

EXCLUDE_WEATHER_FOLLOWUP = ["国家", "首都", "属于", "在哪", "是谁",
                            "叫什么", "人口", "历史", "哪个"]

KNOWLEDGE_HINTS = ["怎么", "如何", "多少", "几天", "流程", "政策",
                   "规定", "是什么", "能不能", "可不可以", "怎么办",
                   "为什么", "几天", "几天假", "标准", "制度"]

REFUSE_TEXT = "这个问题我知识库里没有可靠信息，不敢乱说。你可以补充一下背景，或者换个问法。"


# ============================================================
# 1. 会话上下文
# ============================================================
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
    if session is None or not session.history:
        return text
    if len(text) <= 8 or any(p in text for p in PRONOUNS):
        prev = session.recent_user_texts(1)
        if prev:
            return prev[0] + " " + text
    return text


def is_knowledge_like(text):
    return any(k in text for k in KNOWLEDGE_HINTS)


# ============================================================
# 2. 天气工具
# ============================================================
WMO_ZH = {
    0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}


def geocode_city(city):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
            timeout=6,
        ).json()
        res = r.get("results")
        if not res:
            return None
        g = res[0]
        return g["name"], g["latitude"], g["longitude"]
    except Exception:
        return None


def get_weather(city):
    geo = geocode_city(city)
    if not geo:
        return None
    name, lat, lon = geo
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,"
                           "wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=6,
        ).json()
        cur = r["current"]
        return {
            "city": name, "temp": cur["temperature_2m"],
            "humidity": cur["relative_humidity_2m"],
            "wind": cur["wind_speed_10m"], "code": cur["weather_code"],
        }
    except Exception:
        return None


def is_weather_intent(text):
    return any(k in text for k in WEATHER_KEYS)


def is_weather_followup(text, session=None):
    if session is None or not session.history:
        return False
    prev_users = session.recent_user_texts(2)
    if not prev_users or not any(is_weather_intent(p) for p in prev_users):
        return False
    if any(w in text for w in EXCLUDE_WEATHER_FOLLOWUP):
        return False
    if extract_city(text):
        return len(text) <= 12
    if len(text) <= 8 and any(p in text for p in ["那", "呢", "怎么样", "如何", "多少"]):
        return True
    return False


def extract_city(text):
    known = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉",
             "西安", "南京", "重庆", "天津", "苏州", "长沙", "郑州",
             "沈阳", "青岛", "大连", "厦门", "昆明", "合肥"]
    for c in known:
        if c in text:
            return c
    for key in ["天气", "气温", "温度"]:
        idx = text.find(key)
        if idx > 0:
            prefix = text[:idx].strip("，,。.！!？? ")
            if 1 <= len(prefix) <= 6:
                return prefix
    return None


def extract_city_with_ctx(text, session=None):
    city = extract_city(text)
    if city:
        return city
    if session:
        for prev in reversed(session.recent_user_texts(3)):
            c = extract_city(prev)
            if c:
                return c
    return None


def answer_weather(text, session=None, default_city=DEFAULT_CITY):
    city = extract_city_with_ctx(text, session) or default_city
    info = get_weather(city)
    if not info:
        return f"抱歉，我暂时查不到“{city}”的天气，换个城市名试试？"
    desc = WMO_ZH.get(info["code"], "未知天气")
    return (f"{info['city']}现在{desc}，气温 {info['temp']}℃，"
            f"湿度 {info['humidity']}%，风速 {info['wind']} km/h。")


# ============================================================
# 3. 知识库 RAG（三态检索）
# ============================================================
SAMPLE_KB = """公司年假制度
员工入职满一年享有 5 天带薪年假，满三年享有 10 天，满五年享有 15 天。
年假需提前三个工作日在 OA 系统申请，经直属主管审批后生效。

产品退货政策
自签收之日起 7 天内，商品未拆封可无理由退货；15 天内出现质量问题可换货。
退货需保留原包装和发票，联系客服获取退货地址。

办公时间
公司标准办公时间为周一至周五 9:00-18:00，午休 12:00-13:30。
弹性打卡允许 8:30-10:00 之间到岗，每日工作满 8 小时即可。

差旅报销流程
出差前需在 OA 提交出差申请，回来后 10 个工作日内提交报销单。
交通、住宿需提供发票，餐补按每天 100 元定额发放，无需发票。
"""


def ensure_knowledge_file(path=KNOWLEDGE_FILE):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_KB)
        print(f"[知识库] 未找到 {path}，已生成示例文件")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_chunks(text):
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def char_bigrams(s):
    s = re.sub(r"\s+", "", s)
    if len(s) < 2:
        return [s] if s else []
    return [s[i:i + 2] for i in range(len(s) - 1)]


def build_tfidf(chunks):
    docs = [Counter(char_bigrams(c)) for c in chunks]
    df = Counter()
    for d in docs:
        for term in d:
            df[term] += 1
    N = len(chunks) or 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    vectors = []
    for d in docs:
        vec = {t: tf * idf[t] for t, tf in d.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors, idf


_KB_CHUNKS = []
_KB_VECTORS = []
_KB_IDF = {}


def init_knowledge(path=KNOWLEDGE_FILE):
    global _KB_CHUNKS, _KB_VECTORS, _KB_IDF
    text = ensure_knowledge_file(path)
    _KB_CHUNKS = split_chunks(text)
    _KB_VECTORS, _KB_IDF = build_tfidf(_KB_CHUNKS)
    print(f"[知识库] 加载 {len(_KB_CHUNKS)} 个知识块")


def retrieve(query, top_k=RAG_TOP_K):
    if not _KB_CHUNKS:
        return []
    q = Counter(char_bigrams(query))
    qvec = {t: tf * _KB_IDF.get(t, 1.0) for t, tf in q.items()}
    norm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
    qvec = {t: v / norm for t, v in qvec.items()}

    scored = []
    for i, vec in enumerate(_KB_VECTORS):
        s = sum(qvec.get(t, 0.0) * v for t, v in vec.items())
        scored.append((s, _KB_CHUNKS[i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def answer_rag(query, original_text=None, use_llm=True, history=None):
    """
    返回 (status, answer, context)：
      - "hit"  : 有可靠证据，answer 是回答
      - "weak" : 相关但不够确定，answer 是澄清问句
      - "miss" : 无证据，answer 为 None
    """
    results = retrieve(query)
    if not results:
        return "miss", None, ""

    top_score, top_chunk = results[0]
    if top_score < RAG_SCORE_MISS:
        return "miss", None, ""
    if top_score < RAG_SCORE_HIT:
        title = top_chunk.splitlines()[0] if top_chunk else "相关内容"
        return "weak", f"我不太确定，你是想问“{title}”相关的内容吗？", top_chunk

    context = "\n\n".join(c for s, c in results if s >= RAG_SCORE_MISS)
    question = original_text or query

    if use_llm and _LLM_CLIENT is not None:
        prompt = (
            "只根据下面的资料回答。资料没写的内容，必须回答“资料里没有提到”。"
            "不要推测，不要补充常识。\n\n"
            f"资料：\n{context}\n\n问题：{question}"
        )
        ans = llm_chat(prompt, history=history)
        if ans:
            if ENABLE_VERIFY and not verify_answer(question, context, ans):
                return ("weak",
                        f"我查到了相关资料，但不敢确定回答是否准确。资料原文是：{top_chunk}",
                        top_chunk)
            return "hit", ans, context

    return "hit", f"根据知识库：{top_chunk}", context


# ============================================================
# 4. 本地 Seq2Seq 模型
# ============================================================
DATA = [
    ("你好", "你好呀！我是小智，有什么可以帮你的吗？"),
    ("你是谁", "我是小智，一个对话助手。"),
    ("你叫什么名字", "我叫小智，很高兴认识你。"),
    ("你会做什么", "我可以查天气、查知识库、陪你聊天。"),
    ("谢谢", "不客气，能帮到你就好！"),
    ("再见", "再见，期待下次和你聊天。"),
    ("我很难过", "抱抱你，愿意跟我说说发生了什么吗？"),
    ("我好累", "累了就休息一下吧，身体最重要。"),
    ("推荐一部电影", "可以看看《心灵奇旅》，温暖又治愈。"),
    ("你喜欢什么颜色", "我偏爱蓝色，像深夜的天空。"),
    ("北京是哪个国家的", "北京是中国的首都。"),
    ("1加1等于几", "等于2呀。"),
    ("什么是人工智能", "人工智能就是让机器模拟人类的智能行为。"),
    ("早上好", "早上好！今天也要元气满满哦。"),
    ("晚安", "晚安，做个好梦。"),
    ("我升职了", "恭喜你！这是你努力的结果。"),
    ("讲个笑话", "程序员最讨厌的两件事：写注释，和别人不写注释。"),
    ("你多大了", "我刚被训练出来，还是个宝宝。"),
    ("我想你了", "我也在想你呢，随时欢迎来找我聊天。"),
]


class Vocab:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self):
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def add(self, s):
        for ch in s:
            if ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)

    def enc(self, s, sos=False, eos=False):
        ids = [self.stoi.get(ch, self.UNK) for ch in s]
        if sos:
            ids = [self.SOS] + ids
        if eos:
            ids = ids + [self.EOS]
        return ids

    def dec(self, ids):
        return "".join(self.itos[i] for i in ids if i > self.UNK)

    def __len__(self):
        return len(self.itos)


VOCAB = Vocab()
for a, b in DATA:
    VOCAB.add(a)
    VOCAB.add(b)


class Seq2Seq(nn.Module):
    def __init__(self, vsize, emb=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vsize, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.dec = nn.GRU(emb, hid, batch_first=True)
        self.attn = nn.Linear(hid, hid)
        self.out = nn.Linear(hid * 2, vsize)

    def encode(self, src, lens):
        e = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens, batch_first=True, enforce_sorted=False)
        out, h = self.enc(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, h

    def step(self, token, h, enc_out, mask):
        o, h = self.dec(self.emb(token), h)
        o = o.squeeze(1)
        score = torch.bmm(enc_out, self.attn(o).unsqueeze(2)).squeeze(2)
        score = score.masked_fill(mask == 0, -1e9)
        w = F.softmax(score, dim=1)
        ctx = torch.bmm(w.unsqueeze(1), enc_out).squeeze(1)
        return self.out(torch.cat([o, ctx], dim=1)), h

    def decode(self, tgt_in, enc_out, enc_h, mask):
        h, outs = enc_h, []
        for t in range(tgt_in.size(1)):
            logit, h = self.step(tgt_in[:, t:t + 1], h, enc_out, mask)
            outs.append(logit)
        return torch.stack(outs, dim=1)


def train_step(model, opt, batch):
    src = [torch.tensor(VOCAB.enc(a, eos=True)) for a, _ in batch]
    tgt = [torch.tensor(VOCAB.enc(b, sos=True, eos=True)) for _, b in batch]
    lens = torch.tensor([len(s) for s in src], dtype=torch.long)
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    mask = (src_pad != 0).float()
    enc_out, enc_h = model.encode(src_pad, lens)
    logits = model.decode(tgt_pad[:, :-1], enc_out, enc_h, mask)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_pad[:, 1:].reshape(-1),
        ignore_index=0)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def train_local(model, epochs=300, batch_size=8, lr=1e-3, verbose_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = list(DATA)
    for epoch in range(1, epochs + 1):
        random.shuffle(data)
        total = 0
        for i in range(0, len(data), batch_size):
            total += train_step(model, opt, data[i:i + batch_size])
        if verbose_every and epoch % verbose_every == 0:
            avg = total / ((len(data) + batch_size - 1) // batch_size)
            print(f"  [本地模型] epoch {epoch:3d}  loss={avg:.4f}")


@torch.no_grad()
def local_chat_with_conf(text, model, max_len=40):
    """返回 (answer, confidence)"""
    if not text.strip():
        return "你好像什么都没说，想聊点什么呢？", 1.0
    model.eval()
    src = torch.tensor([VOCAB.enc(text, eos=True)])
    enc_out, h = model.encode(src, torch.tensor([src.size(1)]))
    mask = (src != 0).float()
    cur, ids, probs = torch.tensor([[Vocab.SOS]]), [], []
    for _ in range(max_len):
        logit, h = model.step(cur, h, enc_out, mask)
        p = F.softmax(logit, dim=-1)
        nxt = p.argmax(-1).item()
        if nxt == Vocab.EOS:
            break
        ids.append(nxt)
        probs.append(p.max().item())
        cur = torch.tensor([[nxt]])

    if not ids:
        return "(无输出)", 0.0
    conf = sum(probs) / len(probs)
    ans = VOCAB.dec(ids)
    if conf < LOCAL_CONF_THRESHOLD:
        return "这个问题我不太有把握，换个问法或者告诉我更多背景？", conf
    return ans, conf


# ============================================================
# 5. 大模型后端
# ============================================================
_LLM_CLIENT = None
_LLM_MODEL = None


def init_llm():
    global _LLM_CLIENT, _LLM_MODEL
    if LLM_BACKEND is None:
        return
    try:
        from openai import OpenAI
    except ImportError:
        print("[警告] 未安装 openai，大模型后端不可用。pip install openai")
        return

    if LLM_BACKEND == "deepseek":
        _LLM_CLIENT = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                             base_url="https://api.deepseek.com")
        _LLM_MODEL = "deepseek-chat"
    elif LLM_BACKEND == "openai":
        _LLM_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        _LLM_MODEL = "gpt-4o-mini"


def llm_chat(prompt, history=None):
    if _LLM_CLIENT is None:
        return None

    messages = [{"role": "system",
                 "content": "你是一个友好、简洁的中文助手，回答控制在两三句话内。"}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": prompt})

    try:
        resp = _LLM_CLIENT.chat.completions.create(
            model=_LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[大模型调用失败] {type(e).__name__}: {e}"


def verify_answer(question, context, answer):
    """引用校验：回答是否被资料支持。无大模型时默认通过"""
    if _LLM_CLIENT is None:
        return True
    prompt = (
        "你是事实核查员。判断【回答】中的每一条事实是否都能在【资料】中找到依据。"
        "只输出一个词：SUPPORTED 或 UNSUPPORTED。\n\n"
        f"【资料】\n{context}\n\n【问题】{question}\n\n【回答】{answer}"
    )
    r = llm_chat(prompt)
    return bool(r and "SUPPORTED" in r.upper() and "UNSUPPORTED" not in r.upper())


# ============================================================
# 6. 统一路由
# ============================================================
def chat_response(text, local_model=None, use_llm=True, session=None):
    if not text or not text.strip():
        return "你好像什么都没说，想聊点什么呢？"

    # 1) 天气（含上下文追问）
    if is_weather_intent(text) or is_weather_followup(text, session):
        return answer_weather(text, session=session)

    # 2) 知识库 RAG（三态）
    rag_query = build_rag_query(text, session)
    status, rag_ans, _ = answer_rag(
        rag_query,
        original_text=text,
        use_llm=use_llm,
        history=session.as_messages() if session else None,
    )
    if status == "hit":
        return rag_ans
    if status == "weak":
        return rag_ans
    # status == "miss" 继续往下走

    # 3) 知识类问题但知识库没命中 → 明确拒答
    if is_knowledge_like(text):
        return REFUSE_TEXT

    # 4) 大模型（带历史）
    if use_llm and _LLM_CLIENT is not None:
        ans = llm_chat(text, history=session.as_messages()
                       if session else None)
        if ans:
            return ans

    # 5) 本地 Seq2Seq（带置信度拒答）
    if local_model is not None:
        ans, conf = local_chat_with_conf(text, local_model)
        return ans

    return "我暂时无法回答这个问题。"


# ============================================================
# 7. 测试数据与测试函数
# ============================================================
TEST_WEATHER = ["今天北京天气怎么样", "上海下雨吗", "广州气温多少"]
TEST_RAG = ["年假有几天", "怎么退货", "公司几点上班", "报销流程是什么",
            "出差回来怎么报销", "入职一年有多少天年假"]
TEST_REFUSE = ["火星有多少人口", "量子纠缠是什么", "股票的市盈率怎么算",
               "地球到月球多远", "宇宙的年龄是多少"]
TEST_LOCAL = ["你好", "你是谁", "讲个笑话", "晚安", "我升职了"]
TEST_EDGE = ["", " ", "？？？", "hello", "😀😀😀", "你" * 50]


def run_group(name, questions, model, session=None):
    print(f"\n{'=' * 52}\n{name}\n{'=' * 52}")
    for q in questions:
        q_show = q if len(q) <= 24 else q[:24] + f"...(共{len(q)}字)"
        if not q_show:
            q_show = "(空字符串)"
        try:
            ans = chat_response(q, local_model=model, session=session)
        except Exception as e:
            ans = f"[异常] {type(e).__name__}: {e}"
        print(f"Q: {q_show}")
        print(f"A: {ans}")
        print("-" * 42)


def run_multiturn_test(model):
    print(f"\n{'=' * 52}\nF. 多轮上下文测试\n{'=' * 52}")
    s = ChatSession(max_turns=6)
    turns = [
        "年假有几天",
        "那满三年呢",
        "怎么申请",
        "北京天气怎么样",
        "那上海呢",
        "火星有多少人口",
        "你好",
    ]
    for q in turns:
        ans = chat_response(q, local_model=model, session=s)
        print(f"Q: {q}")
        print(f"A: {ans}")
        print("-" * 42)
        s.add("user", q)
        s.add("assistant", ans)


# ============================================================
# 8. 主流程
# ============================================================
if __name__ == "__main__":
    print(f"词表大小: {len(VOCAB)}")
    print(f"大模型后端: {LLM_BACKEND or '未配置'}")
    print(f"引用校验: {'开启' if ENABLE_VERIFY else '关闭'}")
    print(
        f"RAG 阈值: miss<{RAG_SCORE_MISS}  weak<{RAG_SCORE_HIT}  hit>={RAG_SCORE_HIT}")

    init_llm()
    init_knowledge(KNOWLEDGE_FILE)

    print("\n开始训练本地 Seq2Seq（兜底用）...")
    local_model = Seq2Seq(len(VOCAB), emb=128, hid=256)
    t0 = time.time()
    train_local(local_model, epochs=300, batch_size=8,
                lr=1e-3, verbose_every=100)
    print(f"本地模型训练完成，耗时 {time.time() - t0:.1f}s")

    run_group("A. 天气工具", TEST_WEATHER, local_model)
    run_group("B. 知识库 RAG", TEST_RAG, local_model)
    run_group("C. 知识类拒答（应明确说不知道）", TEST_REFUSE, local_model)
    run_group("D. 本地模型闲聊", TEST_LOCAL, local_model)
    run_group("E. 边界鲁棒性", TEST_EDGE, local_model)
    run_multiturn_test(local_model)

    # ---- 交互模式 ----
    session = ChatSession(max_turns=6)
    print(f"\n{'=' * 52}\n交互模式（输入 '退出' 结束）\n{'=' * 52}")
    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if q in ("退出", "exit", "quit"):
            print("小智: 再见，期待下次和你聊天。")
            break
        if not q:
            continue

        ans = chat_response(q, local_model=local_model, session=session)
        print("小智:", ans)
        session.add("user", q)
        session.add("assistant", ans)
