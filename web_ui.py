import gradio as gr
from router import ChatSession, chat_response
from models.local_seq2seq import train_local
from knowledge_base.rag import init_knowledge
from models.llm_backend import init_llm

# 初始化资源 (全局执行一次)
init_llm()
init_knowledge()
local_model = train_local(epochs=100)

# 每个浏览器会话保持独立的 ChatSession
def respond(message, history):
    # history 由 Gradio 管理，我们可以用它来维持我们自己的 session
    session = ChatSession(max_turns=6)
    # 将 Gradio 的历史记录填充到我们的 session 中，以实现多轮对话
    for human, assistant in history:
        session.add("user", human)
        session.add("assistant", assistant)
    
    # 调用核心路由函数
    ans = chat_response(message, local_model=local_model, session=session)
    
    # 将本轮对话加入历史，供下一轮使用
    session.add("user", message)
    session.add("assistant", ans)
    
    return ans

# 创建 Gradio 聊天界面
demo = gr.ChatInterface(
    fn=respond,
    title="🤖 小智 AI 助手",
    description="集成了本地模型、知识库与工具。",
    examples=["北京天气怎么样", "如果客户嫌贵怎么回", "3加5等于多少"]
)

if __name__ == "__main__":
    demo.launch()