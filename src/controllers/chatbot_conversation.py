import logging
import os

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader

from src.connectors.graph_cove_rag import ChatbotGraphOrchestrator_RAG
from src.connectors.graph_react_judge import ChatbotGraphOrchestrator_ReActJudge
from src.models.chatbot import ChatbotModel
from src.templates.prompt_template import basic_rag_prompt, render_history, render_rag_context

logger = logging.getLogger(__name__)

WORKFLOWS = ("basic", "cove", "react_judge")


def _sources(docs) -> list[str]:
    return sorted({os.path.basename(d.metadata.get("source", "unknown")) for d in docs})


class ChatbotConversationController:
    def __init__(self, data_file: str, data_dir: str = "data", k: int = 5):
        self.data_file = data_file
        self.data_dir = data_dir
        self.k = k

    def generate_response(self, query, chatbot_id, conversation_id, agent, judge, vectorstore, workflow="cove") -> dict:
        if workflow not in WORKFLOWS:
            raise ValueError(f"Unknown workflow {workflow!r}, expected one of {WORKFLOWS}")

        model = ChatbotModel(self.data_file)
        model.load()
        params = model.get_params(chatbot_id)
        if not params:
            raise LookupError(f"Chatbot {chatbot_id} not found")

        conversation_id = str(conversation_id or model.new_conversation_id())
        history = model.get_chat_history(conversation_id)
        info = params.get("assistant_info", {})
        vectorstore.set_collection_name(f"chatbot_{chatbot_id}_collection")

        result: dict = {"workflow": workflow}
        if workflow == "react_judge":
            # The agent decides what to search for itself
            state = ChatbotGraphOrchestrator_ReActJudge(vectorstore, agent.llm, judge.llm, k=self.k + 3).run(query)
            result.update(response=state["response"], judge_score=state["score"], judge_remarks=state["remarks"])
        else:
            docs = vectorstore.similarity_search(query, k=self.k)
            rag_context = render_rag_context(docs)
            result["sources"] = _sources(docs)
            if workflow == "basic":
                prompt = basic_rag_prompt.format(
                    language=info.get("language", "English"),
                    voice_tune=info.get("voice_tune", "Professional"),
                    rag_context=rag_context,
                    chat_history=render_history(history),
                    question=query,
                )
                result["response"] = agent.invoke(prompt)
            else:  # cove
                state = ChatbotGraphOrchestrator_RAG(agent.llm).run(query, rag_context=rag_context)
                result.update(response=state["final_response"], verified=state["is_verified"])

        model.append_message(conversation_id, "human", query)
        model.append_message(conversation_id, "assistant", result["response"])
        model.save()
        result["conversation_id"] = conversation_id
        return result

    def generate_embeddings(self, chatbot_id, vectorstore) -> int:
        model = ChatbotModel(self.data_file)
        model.load()
        params = model.get_params(chatbot_id)
        if not params:
            raise LookupError(f"Chatbot {chatbot_id} not found")
        uploaded_files = params.get("uploaded_files", [])
        if not uploaded_files:
            raise ValueError("No uploaded_files configured for this chatbot")

        vectorstore.set_collection_name(f"chatbot_{chatbot_id}_collection")
        all_chunks = []
        for file_name in uploaded_files:
            path = os.path.join(self.data_dir, file_name)
            ext = os.path.splitext(path)[1].lower()
            if ext in {".txt", ".md"}:
                loader = TextLoader(path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyMuPDFLoader(path, mode="single")
            elif ext == ".csv":
                loader = CSVLoader(path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            all_chunks.extend(vectorstore.split_documents(loader.load()))

        vectorstore.add_documents(all_chunks)
        logger.info("Indexed %d chunks for chatbot %s", len(all_chunks), chatbot_id)
        return len(all_chunks)
