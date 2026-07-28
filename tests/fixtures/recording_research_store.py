"""Reusable in-memory store that records Notebook lifecycle call order."""
from tether.context.memory_store import MemoryStore


class RecordingResearchStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    async def start_turn(self, session_id, turn_id, *, model_name=None):
        self.calls.append("start_turn")
        await super().start_turn(session_id, turn_id, model_name=model_name)

    async def get_history(self, session_id, include_thinking=False):
        self.calls.append("get_history")
        return await super().get_history(session_id, include_thinking)

    async def add_user(self, session_id, text, *, turn_id=None, seq_start=None):
        self.calls.append("add_user")
        await super().add_user(session_id, text, turn_id=turn_id, seq_start=seq_start)

    async def add_assistant_text(self, session_id, text, thinking_text=None, save_thinking=True, *, turn_id=None, seq_start=None):
        self.calls.append("add_assistant_text")
        await super().add_assistant_text(session_id, text, thinking_text, save_thinking, turn_id=turn_id, seq_start=seq_start)

    async def complete_turn(self, turn_id, *, status="completed", stop_reason=None, error_json=None):
        self.calls.append("complete_turn")
        await super().complete_turn(turn_id, status=status, stop_reason=stop_reason, error_json=error_json)
