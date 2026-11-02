import os


class AppConfig:
    """All settings come from environment variables (see .env.example)."""

    def __init__(self):
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
        self.DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
        self.PORT = int(os.environ.get("PORT", "5125"))

        # Azure OpenAI
        self.AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        self.AZURE_OPENAI_JUDGE_DEPLOYMENT = os.environ.get(
            "AZURE_OPENAI_JUDGE_DEPLOYMENT", self.AZURE_OPENAI_CHAT_DEPLOYMENT
        )
        self.AZURE_OPENAI_CHAT_API_VERSION = os.environ.get("AZURE_OPENAI_CHAT_API_VERSION", "2025-01-01-preview")
        self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )
        self.AZURE_OPENAI_EMBEDDING_API_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15")

        # Vector store (pgvector)
        self.POSTGRES_CONNECTION_STRING = os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        )

        # Which RAG workflow answers questions: basic | cove | react_judge
        self.RAG_WORKFLOW = os.environ.get("RAG_WORKFLOW", "cove")
        self.RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))

        # Local storage for bot settings and conversation history
        self.CHATBOT_DATA_FILE = os.environ.get("CHATBOT_DATA_FILE", "src/chatbot_data.json")
        self.DATA_DIR = os.environ.get("DATA_DIR", "data")
