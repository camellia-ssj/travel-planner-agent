"""旅行规划 Agent 包。"""

__all__ = ["__version__"]

__version__ = "0.1.0"

# 对话图采用延迟导入以避免模块加载时的循环引用。
# 使用方式:
#     from travel_agent.conversation import build_conversation_graph, ConversationState
