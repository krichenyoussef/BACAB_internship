import logging
from functools import lru_cache

import tiktoken
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens with OpenAI's tokenizer, so chunk sizes match model limits."""
    return len(_encoding().encode(text))


class VectorStoreConnector:
    """pgvector store with one collection per chatbot."""

    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        azure_endpoint: str,
        openai_api_key: str,
        embedding_deployment: str = "text-embedding-3-small",
        embedding_api_version: str = "2023-05-15",
        embeddings=None,
    ):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = embeddings or AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            api_version=embedding_api_version,
            azure_endpoint=azure_endpoint,
            api_key=openai_api_key,
            dimensions=1536,
            chunk_size=2048,  # texts embedded per batch
        )
        self.vector_store = self._build_store()

    def _build_store(self) -> PGVector:
        return PGVector(
            collection_name=self.collection_name,
            connection=self.connection_string,
            embeddings=self.embeddings,
            pre_delete_collection=False,
            use_jsonb=True,
            embedding_length=1536,
        )

    def set_collection_name(self, collection_name: str) -> None:
        if collection_name != self.collection_name:
            self.collection_name = collection_name
            self.vector_store = self._build_store()

    @staticmethod
    def split_documents(docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
        """Token-based chunking (sizes are in tokens, not characters)."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=count_tokens
        )
        return splitter.split_documents(docs)

    def add_documents(self, docs: list[Document]):
        return self.vector_store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        docs = self.vector_store.similarity_search(query, k=k)
        logger.debug("Retrieved %d chunks for query %r", len(docs), query)
        return docs
