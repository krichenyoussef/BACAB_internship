import os
import faiss
import pickle
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Azure config depuis .env
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # e.g. "2024-02-15-preview"
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")  # e.g. "text-embedding-3-small"

# Chemin de sauvegarde
INDEX_PATH = "mycode/vector_index.faiss"
DOCS_PATH = "mycode/vector_docs.pkl"

# Charger ou créer l'index FAISS
def create_or_load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(1536)  # text-embedding-3-small returns 1536-dim vectors
        documents = []
    return index, documents

# Obtenir les embeddings via Azure
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=DEPLOYMENT_NAME,
        input=texts
    )
    return [item.embedding for item in response.data]

# Ajouter des documents
def add_documents(docs: list[str]):
    index, existing_docs = create_or_load_index()
    vectors = get_embeddings(docs)
    index.add(np.array(vectors).astype("float32"))
    all_docs = existing_docs + docs
    save_index(index, all_docs)
    return len(all_docs)

# Sauvegarder l’index et les docs
def save_index(index, docs):
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

# Rechercher dans l’index
def search(query: str, k=3):
    index, docs = create_or_load_index()
    if index.ntotal == 0:
        return []
    query_vector = get_embeddings([query])
    distances, indices = index.search(np.array(query_vector).astype("float32"), k)
    return [docs[i] for i in indices[0] if i < len(docs)]
