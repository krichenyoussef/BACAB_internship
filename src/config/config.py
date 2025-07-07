import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    # Azure OpenAI Config
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
    OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # nom du *deployment* Azure
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))

    # RAG
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/faiss_index")
