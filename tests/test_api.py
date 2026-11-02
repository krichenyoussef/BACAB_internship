import json


def test_health_check(make_client):
    assert make_client([]).get("/health_check").status_code == 200


def test_ask_basic_returns_answer_sources_and_saves_history(make_client, data_file):
    client = make_client(["You get 25 days of paid leave per year (source: company_handbook.md)."])
    resp = client.post("/chatbot_api/v1/chatbot/1/ask", json={"query": "How many leave days?", "workflow": "basic"})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["response"].startswith("You get 25 days")
    assert body["sources"] == ["company_handbook.md"]

    saved = json.load(open(data_file))
    turns = saved["conversations"][body["conversation_id"]]
    assert [t["role"] for t in turns] == ["human", "assistant"]


def test_ask_continues_existing_conversation(make_client):
    client = make_client(["First answer.", "Second answer."])
    first = client.post("/chatbot_api/v1/chatbot/1/ask", json={"query": "Q1", "workflow": "basic"}).get_json()
    second = client.post(
        "/chatbot_api/v1/chatbot/1/ask",
        json={"query": "Q2", "workflow": "basic", "conversation_id": first["conversation_id"]},
    ).get_json()
    assert second["conversation_id"] == first["conversation_id"]


def test_ask_cove_workflow(make_client):
    client = make_client(
        [
            "<relevant_section>25 days of paid annual leave.</relevant_section>",
            "25 days per year.",
            "NO VERIFICATION NEEDED",
        ]
    )
    body = client.post("/chatbot_api/v1/chatbot/1/ask", json={"query": "Leave days?", "workflow": "cove"}).get_json()
    assert body["response"] == "25 days per year."
    assert body["verified"] is True


def test_ask_validates_input(make_client):
    client = make_client([])
    assert client.post("/chatbot_api/v1/chatbot/1/ask", json={}).status_code == 400
    assert client.post("/chatbot_api/v1/chatbot/1/ask", json={"query": "hi", "workflow": "nope"}).status_code == 400


def test_ask_unknown_chatbot_returns_404(make_client):
    resp = make_client(["unused"]).post("/chatbot_api/v1/chatbot/99/ask", json={"query": "hi", "workflow": "basic"})
    assert resp.status_code == 404
