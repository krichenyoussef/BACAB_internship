from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.memory import RedisChatMessageHistory
from langchain.chains import ConversationChain
from config.config import settings  # ex: chargement API_KEY depuis .env

def get_agent_with_memory(redis_url: str):
    memory = RedisChatMessageHistory(
        session_id="session_1",
        url=redis_url
    )

    llm = OpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY)

    tools = [
        Tool(
            name="SimpleTool",
            func=lambda x: "Outil fictif exécuté avec: " + x,
            description="Exemple de Tool LangChain"
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-zero-shot-react-description",
        memory=memory,
        verbose=True
    )

    return agent