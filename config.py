# config.py
import os
import sys

# 判断是否是打包后的 EXE 运行环境
if getattr(sys, 'frozen', False):
    # 如果是 EXE，将工作目录切换到 EXE 所在的文件夹
    os.chdir(os.path.dirname(sys.executable))

DEFAULT_CITY = "北京"
KNOWLEDGE_FILE = "knowledge.txt"
LOCAL_MODEL_PATH = "seq2seq_model.pt"  # 模型保存路径

RAG_TOP_K = 3
RAG_SCORE_MISS_TFIDF = 0.10
RAG_SCORE_HIT_TFIDF = 0.25
RAG_SCORE_MISS_VEC = 0.50
RAG_SCORE_HIT_VEC = 0.62

USE_VECTOR_RETRIEVAL = True
VECTOR_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

ENABLE_VERIFY = False
LOCAL_CONF_THRESHOLD = 0.5

LLM_BACKEND = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else (
    "openai" if os.environ.get("OPENAI_API_KEY") else None
)

WEATHER_KEYS = ["天气", "气温", "温度", "下雨", "下雪", "刮风", "冷不冷", "热不热", "穿什么"]
TIME_KEYS = ["几点", "时间", "几号", "日期", "星期几", "周几", "今天几"]
WRITING_KEYS = ["写", "起草", "邮件", "文案", "诗", "文章", "作文", "总结", "大纲"]
PRONOUNS = ["它", "他", "她", "这个", "那个", "这", "那", "再", "还", "另外", "上面", "前面", "呢"]
EXCLUDE_WEATHER_FOLLOWUP = ["国家", "首都", "属于", "在哪", "是谁", "叫什么", "人口", "历史", "哪个", "钱", "价格", "元", "买", "库存", "功能", "特点", "怎么卖"]
KNOWLEDGE_HINTS = ["怎么", "如何", "多少", "几天", "流程", "政策", "规定", "是什么", "能不能", "可不可以", "怎么办", "为什么", "标准", "制度", "话术", "价格", "库存", "参数", "特点", "推荐"]

REFUSE_TEXT = "这个问题我知识库里没有可靠信息，不敢乱说。你可以补充一下背景，或者换个问法。"