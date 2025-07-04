import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from rag_search_tool import rag_search_tool

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

chat_history = RedisChatMessageHistory(session_id="main-session", url=os.getenv("REDIS_URL"))
memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)

tools = [rag_search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="chat-conversational-react-description",
    verbose=True
)
