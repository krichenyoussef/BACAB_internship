"""Shared fakes: no Azure account or database needed to run the tests."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.app import create_app
from src.config.config import AppConfig
from src.connectors.llm_connector import AzureChatbotAgent

HANDBOOK_CHUNKS = [
    Document(
        page_content="Every full-time employee receives 25 days of paid annual leave per year.",
        metadata={"source": "data/sample/company_handbook.md"},
    ),
    Document(
        page_content="Leave requests must be submitted at least 10 working days in advance.",
        metadata={"source": "data/sample/company_handbook.md"},
    ),
]


class FakeVectorStore:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else HANDBOOK_CHUNKS
        self.collection_name = None
        self.queries = []

    def set_collection_name(self, name):
        self.collection_name = name

    def similarity_search(self, query, k=5):
        self.queries.append(query)
        return self.docs[:k]


def fake_agent(responses):
    return AzureChatbotAgent(openai_api_key="x", azure_endpoint="x", llm=FakeListChatModel(responses=responses))


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "chatbot_data.json"
    path.write_text(
        json.dumps(
            {
                "chatbots": {"1": {"name": "Test bot", "params": {"assistant_info": {"language": "English"}}}},
                "conversations": {},
            }
        )
    )
    return str(path)


@pytest.fixture
def make_client(data_file):
    def _make(agent_responses, judge_responses=("{}",), store=None):
        cfg = AppConfig()
        cfg.CHATBOT_DATA_FILE = data_file
        app = create_app(cfg, init_connectors=False)
        app.azure_chatbot_agent_connector = fake_agent(list(agent_responses))
        app.judge_agent_connector = fake_agent(list(judge_responses))
        app.vectorstore_connector = store or FakeVectorStore()
        return app.test_client()

    return _make
