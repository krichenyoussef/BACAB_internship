# src/services/rag_service.py

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from src.config.config import AppConfig
import os

# Initialisation des embeddings OpenAI
embedding_model = OpenAIEmbeddings(openai_api_key=AppConfig.OPENAI_API_KEY)

# Chargement du vecteur store FAISS
def load_vectorstore():
    if not os.path.exists(AppConfig.VECTOR_STORE_PATH):
        raise FileNotFoundError("Index FAISS introuvable à l'emplacement spécifié.")
    return FAISS.load_local(AppConfig.VECTOR_STORE_PATH, embedding_model)

def get_rag_response(question: str) -> str:
    """
    Utilise LangChain + FAISS pour générer une réponse basée sur le contexte vectoriel.
    """
    try:
        # Chargement du vecteur store FAISS
        vectorstore = load_vectorstore()

        # Création du modèle de récupération + génération
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=AppConfig.OPENAI_API_KEY,
                model_name=AppConfig.OPENAI_MODEL_NAME,
                temperature=0.3
            ),
            retriever=retriever,
            return_source_documents=False
        )

        result = qa_chain.run(question)
        return f"[RAG] {result}"

    except Exception as e:
        return f"[RAG - ERREUR] {str(e)}"
