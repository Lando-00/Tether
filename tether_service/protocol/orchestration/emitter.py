import json
from datetime import datetime, timezone
from typing import Dict, Any

class NdjsonEmitter:
    """Emitter producing NDJSON bytes for unified event schema"""
    def emit(self, event: Dict[str, Any]) -> bytes:
        # A4 PAIN-3: use timezone-aware datetime to produce unambiguous ISO 8601
        # with +00:00 suffix instead of naive UTC (deprecated in Python 3.12).
        out = {
            "type": event.get("type", ""),
            "session_id": event.get("session_id", ""),
            "data": event.get("data", {}),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        return (json.dumps(out) + "\n").encode("utf-8")