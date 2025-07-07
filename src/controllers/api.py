# src/controllers/api.py
from flask import Blueprint, request, jsonify
from src.agents.chatbot import handle_question

api_bp = Blueprint("api", __name__)

@api_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    confidence = data.get("confidence_threshold", 70)

    answer, mode = handle_question(question, confidence)
    return jsonify({
        "answer": answer,
        "mode": mode
    })
