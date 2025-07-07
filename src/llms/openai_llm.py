# src/llms/openai_llm.py

from langchain.chat_models import ChatOpenAI
from src.config.config import AppConfig

def get_llm() -> ChatOpenAI:
    """
    Initialise un LLM LangChain compatible avec OpenAI ou Azure.
    """
    return ChatOpenAI(
        model_name=AppConfig.OPENAI_MODEL_NAME,
        temperature=AppConfig.LLM_TEMPERATURE,
        openai_api_key=AppConfig.OPENAI_API_KEY
    )
