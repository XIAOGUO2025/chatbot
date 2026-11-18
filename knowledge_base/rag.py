# knowledge_base/rag.py
import math
import os
import re
from collections import Counter
from config import (KNOWLEDGE_FILE, RAG_TOP_K, RAG_SCORE_MISS_TFIDF, RAG_SCORE_HIT_TFIDF,
                    RAG_SCORE_MISS_VEC, RAG_SCORE_HIT_VEC, USE_VECTOR_RETRIEVAL, VECTOR_MODEL_NAME)
from models.llm_backend import get_llm_client, llm_chat

SAMPLE_KB = """【话术：欢迎语】
您好！欢迎光临，我是您的专属客服小智，请问有什么可以帮您？

【话术：缺货安抚】
非常抱歉，您看中的这款商品目前暂时缺货了。我们已经紧急补货，预计3天后到货。您看是先给您推荐同类产品，还是等几天给您优先发货呢？

【物品：降噪耳机 Pro】
价格：899元
库存：现货充足
特点：主动降噪45dB，超长续航40小时，支持无线充电，佩戴轻盈舒适。

【物品：智能手表 X1】
价格：1299元
库存：仅剩3件
特点：血氧心率监测，支持100种运动模式，5ATM防水，高清彩屏。
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
    if len(s) < 2: return [s] if s else []
    return [s[i:i + 2] for i in range(len(s) - 1)]

def build_tfidf(chunks):
    docs = [Counter(char_bigrams(c)) for c in chunks]
    df = Counter()
    for d in docs:
        for term in d: df[term] += 1
    N = len(chunks) or 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
    vectors = []
    for d in docs:
        vec = {t: tf * idf[t] for t, tf in d.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors, idf

_KB_CHUNKS, _KB_VECTORS, _KB_IDF = [], [], {}
_VEC_MODEL, _FAISS_INDEX, _VEC_DIM, _VEC_FAILED = None, None, None, False

def init_vector_backend():
    global _VEC_MODEL, _FAISS_INDEX, _VEC_DIM, _VEC_FAILED
    if not USE_VECTOR_RETRIEVAL or _VEC_FAILED: return False
    if _VEC_MODEL is not None and _FAISS_INDEX is not None: return True
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[向量检索] 未安装相关库，回退 TF-IDF"); _VEC_FAILED = True; return False
    try:
        print(f"[向量检索] 正在加载模型 {VECTOR_MODEL_NAME} ...")
        _VEC_MODEL = SentenceTransformer(VECTOR_MODEL_NAME)
        _VEC_DIM = _VEC_MODEL.get_sentence_embedding_dimension()
        _FAISS_INDEX = faiss.IndexFlatIP(_VEC_DIM)
        return True
    except Exception as e:
        print(f"[向量检索] 加载失败: {e}"); _VEC_MODEL = None; _VEC_FAILED = True; return False

def rebuild_vector_index():
    global _FAISS_INDEX
    if not init_vector_backend() or not _KB_CHUNKS: return
    import faiss; import numpy as np
    vecs = _VEC_MODEL.encode(_KB_CHUNKS, normalize_embeddings=True, show_progress_bar=False)
    _FAISS_INDEX = faiss.IndexFlatIP(_VEC_DIM)
    _FAISS_INDEX.add(np.asarray(vecs, dtype="float32"))

def init_knowledge(path=KNOWLEDGE_FILE):
    global _KB_CHUNKS, _KB_VECTORS, _KB_IDF
    text = ensure_knowledge_file(path)
    _KB_CHUNKS = split_chunks(text)
    _KB_VECTORS, _KB_IDF = build_tfidf(_KB_CHUNKS)
    print(f"[知识库] 加载 {len(_KB_CHUNKS)} 个知识块")
    rebuild_vector_index()

def retrieve(query, top_k=RAG_TOP_K):
    if not _KB_CHUNKS: return []
    if USE_VECTOR_RETRIEVAL and init_vector_backend():
        if _FAISS_INDEX is None or _FAISS_INDEX.ntotal == 0: rebuild_vector_index()
        if _VEC_MODEL and _FAISS_INDEX and _FAISS_INDEX.ntotal > 0:
            import numpy as np
            qv = _VEC_MODEL.encode([query], normalize_embeddings=True, show_progress_bar=False)
            scores, idxs = _FAISS_INDEX.search(np.asarray(qv, dtype="float32"), top_k)
            out = []
            for s, i in zip(scores[0], idxs[0]):
                if i >= 0: out.append((float(s), _KB_CHUNKS[i]))
            if out: return out
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

def get_rag_thresholds():
    if USE_VECTOR_RETRIEVAL and _VEC_MODEL is not None: return RAG_SCORE_MISS_VEC, RAG_SCORE_HIT_VEC
    return RAG_SCORE_MISS_TFIDF, RAG_SCORE_HIT_TFIDF

def answer_rag(query, original_text=None, use_llm=True, history=None):
    results = retrieve(query)
    if not results: return "miss", None, ""
    
    question = original_text or query
    
    # 【强匹配重排】：防止向量模型犯傻
    for i in range(len(results)):
        score, chunk = results[i]
        first_line = chunk.splitlines()[0] if chunk else ""
        
        # 提取标题里的有效汉字
        clean_title = re.sub(r'[【】\[\]:：话术物品]', '', first_line)
        
        match_count = 0
        for j in range(len(clean_title) - 1):
            if clean_title[j:j+2] in question:
                match_count += 1
                
        if match_count > 0:
            results[i] = (score + 0.8 * match_count, chunk)

    # 重新按分数排序
    results.sort(key=lambda x: x[0], reverse=True)
    
    miss_th, hit_th = get_rag_thresholds()
    top_score, top_chunk = results[0]
    
    # 低于阈值直接当作没查到，不反问
    if top_score < hit_th: 
        return "miss", None, ""
        
    context = "\n\n".join(c for s, c in results if s >= miss_th)
    
    client, _ = get_llm_client()
    if use_llm and client is not None:
        prompt = f"只根据下面的资料回答。资料没写的内容，必须回答“资料里没有提到”。不要推测。\n\n资料：\n{context}\n\n问题：{question}"
        ans = llm_chat(prompt, history=history)
        if ans: return "hit", ans, context
    return "hit", f"根据知识库：{top_chunk}", context