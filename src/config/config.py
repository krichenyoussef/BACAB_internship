# src/config/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Charge les variables depuis le .env
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    REDIS_URL: str = os.getenv("REDIS_URL")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "openai")  # default openai
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/index.faiss")

settings = Settings()
