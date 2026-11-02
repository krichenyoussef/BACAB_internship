import logging

from flask import Blueprint, current_app, jsonify, request

from src.controllers.chatbot_conversation import WORKFLOWS, ChatbotConversationController

chatbot_conversation_bp = Blueprint("chatbot_conversation", __name__)
logger = logging.getLogger(__name__)


def _controller() -> ChatbotConversationController:
    cfg = current_app.config
    return ChatbotConversationController(cfg["CHATBOT_DATA_FILE"], cfg["DATA_DIR"], cfg["RETRIEVAL_K"])


@chatbot_conversation_bp.post("/chatbot/<int:chatbot_id>/ask")
def ask(chatbot_id):
    """Body: {"query": "...", "conversation_id": "1" (optional), "workflow": "basic|cove|react_judge" (optional)}"""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "'query' is required"}), 400
    workflow = data.get("workflow", current_app.config["RAG_WORKFLOW"])
    if workflow not in WORKFLOWS:
        return jsonify({"error": f"'workflow' must be one of {list(WORKFLOWS)}"}), 400

    try:
        result = _controller().generate_response(
            query=query,
            chatbot_id=chatbot_id,
            conversation_id=data.get("conversation_id"),
            agent=current_app.azure_chatbot_agent_connector,
            judge=current_app.judge_agent_connector,
            vectorstore=current_app.vectorstore_connector,
            workflow=workflow,
        )
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except Exception:
        logger.exception("Failed to generate a response")
        return jsonify({"error": "Could not generate a response"}), 500
    return jsonify(result), 200


@chatbot_conversation_bp.post("/chatbot/<int:chatbot_id>/generate_embeddings")
def generate_embeddings(chatbot_id):
    try:
        n_chunks = _controller().generate_embeddings(chatbot_id, current_app.vectorstore_connector)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        logger.exception("Failed to generate embeddings")
        return jsonify({"error": "Could not generate embeddings"}), 500
    return jsonify({"status": "ok", "chunks_indexed": n_chunks}), 200
