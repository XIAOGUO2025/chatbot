# knowledge_base/document_loader.py
import os
import re
from config import KNOWLEDGE_FILE
from knowledge_base.rag import split_chunks, init_knowledge

def extract_text_from_file(path):
    if not os.path.exists(path): return None
    lower = path.lower()
    try:
        if lower.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        if lower.endswith(".docx"):
            from docx import Document
            return "\n".join(p.text for p in Document(path).paragraphs)
        if lower.endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8") as f: return f.read()
    except Exception as e:
        print(f"[文档解析失败] {e}")
        return None
    return None

def clean_doc_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def upload_document(path, source_name=None):
    if not os.path.exists(path): return f"文件不存在：{path}"
    text = extract_text_from_file(path)
    if not text or not text.strip(): return f"没能从 {path} 提取到文本。"
    name = source_name or os.path.basename(path)
    
    chunks = split_chunks(clean_doc_text(text))
    if not chunks: return "文档内容为空。"
    
    with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n")
        for c in chunks: f.write(f"【来源：{name}】\n{c}\n\n")
    
    init_knowledge(KNOWLEDGE_FILE)
    return f"已从 {name} 导入 {len(chunks)} 个知识块。"