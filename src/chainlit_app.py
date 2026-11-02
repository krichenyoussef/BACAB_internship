"""Chat UI for manual testing. Run the API first, then:
    chainlit run src/chainlit_app.py --port 5000
"""
import os

import chainlit as cl
import requests

API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:5125/chatbot_api/v1")
CHATBOT_ID = int(os.environ.get("CHATBOT_ID", "1"))


@cl.on_chat_start
async def start():
    cl.user_session.set("conversation_id", None)
    await cl.Message(
        content="Ask a question about the indexed documents.\n"
        "Type `/workflow basic|cove|react_judge` to switch the RAG workflow."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    text = message.content.strip()
    if text.startswith("/workflow"):
        workflow = text.split(maxsplit=1)[-1].strip()
        cl.user_session.set("workflow", workflow)
        await cl.Message(content=f"Workflow set to `{workflow}`.").send()
        return

    payload = {"query": text, "conversation_id": cl.user_session.get("conversation_id")}
    if cl.user_session.get("workflow"):
        payload["workflow"] = cl.user_session.get("workflow")

    try:
        resp = requests.post(f"{API_URL}/chatbot/{CHATBOT_ID}/ask", json=payload, timeout=120)
        data = resp.json()
    except Exception as e:
        await cl.Message(content=f"API error: {e}").send()
        return

    if resp.status_code != 200:
        await cl.Message(content=f"API error {resp.status_code}: {data.get('error')}").send()
        return

    cl.user_session.set("conversation_id", data["conversation_id"])
    details = []
    if data.get("sources"):
        details.append("Sources: " + ", ".join(data["sources"]))
    if data.get("verified") is not None:
        details.append(f"Verified: {data['verified']}")
    if data.get("judge_score") is not None:
        details.append(f"Judge score: {data['judge_score']}/10")
    footer = f"\n\n_{' · '.join(details)}_" if details else ""
    await cl.Message(content=data["response"] + footer).send()
