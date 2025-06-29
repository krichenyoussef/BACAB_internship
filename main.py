from flask import Flask, request, jsonify
from mycode.rag_indexer import add_documents, search
from agent_with_redis import agent

app = Flask(__name__)

@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json()
    docs = data.get("documents", [])
    if not docs:
        return jsonify({"error": "documents missing"}), 400
    count = add_documents(docs)
    return jsonify({"indexed": count})

@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "query missing"}), 400
    results = search(query)
    return jsonify({"results": results})

@app.route("/agent", methods=["POST"])
def agent_query():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "question missing"}), 400
    answer = agent.run(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
