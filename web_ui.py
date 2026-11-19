import streamlit as st
import time
from router import ChatSession, chat_response
from models.local_seq2seq import train_local
from knowledge_base.rag import init_knowledge
from models.llm_backend import init_llm

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(page_title="小智 AI 助手", page_icon="🤖", layout="centered")
st.title("🤖 小智 AI 助手")
st.caption("集成了本地大模型、向量检索知识库与实用工具")

# ==========================================
# 侧边栏（设置与状态）
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    # 清空对话按钮
    if st.button("🧹 清空对话记录"):
        st.session_state.messages = []
        st.session_state.session = ChatSession(max_turns=6)
        st.rerun()
        
    st.divider()
    st.markdown("**当前能力:**")
    st.markdown("- 🕒 时间与科学计算\n- ☀️ 实时天气\n- 📚 知识库 RAG\n- 💬 本地模型闲聊")
    
    if st.session_state.get("llm_loaded", False):
        st.success("✅ 大模型已连接")
    else:
        st.warning("⚠️ 未配置大模型 Key，部分功能（写作等）不可用。")

# ==========================================
# 初始化缓存（只运行一次，避免每次对话重新加载模型）
# ==========================================
@st.cache_resource(show_spinner="正在加载知识库与本地模型，请稍候...")
def init_app():
    init_llm()
    init_knowledge()
    model = train_local(epochs=100) # 如果你之前训练过，这里会秒加载
    return model

local_model = init_app()
st.session_state.llm_loaded = False # 根据实际情况修改

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session" not in st.session_state:
    st.session_state.session = ChatSession(max_turns=6)

# ==========================================
# 渲染历史对话
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 接收用户输入并处理
# ==========================================
if prompt := st.chat_input("输入你的问题，例如：北京天气怎么样 / 键盘有现货吗"):
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 显示 AI 思考动画并获取回答
    with st.chat_message("assistant"):
        with st.spinner("小智正在思考..."):
            # 调用我们之前写好的路由
            ans = chat_response(prompt, local_model=local_model, session=st.session_state.session)
            
            # 模拟打字机效果（可选，让体验更好）
            message_placeholder = st.empty()
            full_response = ""
            for chunk in ans.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(ans)
            
    # 3. 保存记录
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.session_state.session.add("user", prompt)
    st.session_state.session.add("assistant", ans)