import json
import os
import threading
from typing import Any, Dict, List

_LOCK = threading.Lock()


class ChatbotModel:
    """Bot settings and conversation history stored in a JSON file.

    Good enough for a demo; a real deployment would use a database
    (the file is rewritten on every message).
    """

    def __init__(self, filename: str):
        self.filename = filename
        self._data: Dict[str, Any] = {}

    def load(self) -> None:
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except FileNotFoundError:
            self._data = {}
        self._data.setdefault("chatbots", {})
        self._data.setdefault("conversations", {})

    def save(self) -> None:
        with _LOCK:  # atomic write: temp file, then replace
            tmp = self.filename + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, self.filename)

    def get_name(self, bot_id) -> str:
        return self._data["chatbots"][str(bot_id)]["name"]

    def get_params(self, bot_id) -> Dict[str, Any]:
        return self._data["chatbots"].get(str(bot_id), {}).get("params", {})

    def get_chat_history(self, conversation_id) -> List[Dict[str, str]]:
        return self._data["conversations"].get(str(conversation_id), [])

    def append_message(self, conversation_id, role: str, content: str) -> None:
        self._data["conversations"].setdefault(str(conversation_id), []).append({"role": role, "content": content})

    def new_conversation_id(self) -> str:
        ids = [int(c) for c in self._data["conversations"] if str(c).isdigit()]
        return str(max(ids, default=0) + 1)
