from src.connectors.llm_connector import AzureChatbotAgent
from src.connectors.vectorstore_connector import VectorStoreConnector


class AppContext:
    @classmethod
    def init_context(cls, app):
        """Create the connectors once and share them between controllers."""
        cfg = app.config
        with app.app_context():
            app.azure_chatbot_agent_connector = AzureChatbotAgent(
                openai_api_key=cfg["AZURE_OPENAI_API_KEY"],
                azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
                deployment_name=cfg["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                api_version=cfg["AZURE_OPENAI_CHAT_API_VERSION"],
            )
            app.judge_agent_connector = AzureChatbotAgent(
                openai_api_key=cfg["AZURE_OPENAI_API_KEY"],
                azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
                deployment_name=cfg["AZURE_OPENAI_JUDGE_DEPLOYMENT"],
                api_version=cfg["AZURE_OPENAI_CHAT_API_VERSION"],
            )
            app.vectorstore_connector = VectorStoreConnector(
                connection_string=cfg["POSTGRES_CONNECTION_STRING"],
                collection_name="chatbot_1_collection",
                azure_endpoint=cfg["AZURE_OPENAI_ENDPOINT"],
                openai_api_key=cfg["AZURE_OPENAI_API_KEY"],
                embedding_deployment=cfg["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                embedding_api_version=cfg["AZURE_OPENAI_EMBEDDING_API_VERSION"],
            )
