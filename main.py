from src.agents.langchain_agent import get_agent_with_memory

agent = get_agent_with_memory(redis_url="redis://localhost:6379")

response = agent.run("Quel est l'état du système ?")
print(response)
