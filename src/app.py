from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

from src.app_context import AppContext
from src.config.config import AppConfig
from src.routes.chatbot_conversation import chatbot_conversation_bp
from src.utils.logging import create_logger

load_dotenv()


def create_app(config: AppConfig | None = None, init_connectors: bool = True) -> Flask:
    application = Flask(__name__)
    application.config.from_object(config or AppConfig())

    if init_connectors:
        # Shared connectors (LLM, vector store) live on the app object
        AppContext.init_context(application)

    application.register_blueprint(chatbot_conversation_bp, url_prefix="/chatbot_api/v1")

    @application.get("/health_check")
    def health_check():
        return jsonify({"status": "ok"}), 200

    CORS(application)
    return application


if __name__ == "__main__":
    app = create_app()
    logger = create_logger(app.config["LOG_LEVEL"])
    logger.info("Chatbot API started (workflow: %s)", app.config["RAG_WORKFLOW"])
    app.run(host="0.0.0.0", port=app.config["PORT"], debug=app.config["DEBUG"])
