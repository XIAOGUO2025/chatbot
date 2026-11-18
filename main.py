# main.py
import time
from models.llm_backend import init_llm
from models.local_seq2seq import train_local
from knowledge_base.rag import init_knowledge
from knowledge_base.document_loader import upload_document
from router import ChatSession, chat_response

if __name__ == "__main__":
    print("=== 正在初始化环境 ===")
    init_llm()
    init_knowledge()
    
    print("\n=== 准备本地 Seq2Seq（兜底用） ===")
    t0 = time.time()
    local_model = train_local(epochs=100)
    print(f"本地模型准备完成，耗时 {time.time() - t0:.1f}s")

    session = ChatSession(max_turns=6)
    print("\n" + "=" * 50)
    print("交互模式已启动")
    print("  - 直接输入问题开始聊天")
    print("  - 输入 '上传 文件路径' 导入文档")
    print("  - 输入 '退出' / 'exit' 结束")
    print("=" * 50)
    
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
            
        if q in ("退出", "exit", "quit"):
            print("小智: 再见，期待下次和你聊天。")
            break
        if not q:
            continue

        if q.startswith("上传 "):
            path = q[3:].strip()
            print("小智:", upload_document(path))
            continue

        ans = chat_response(q, local_model=local_model, session=session)
        print("小智:", ans)
        
        session.add("user", q)
        session.add("assistant", ans)