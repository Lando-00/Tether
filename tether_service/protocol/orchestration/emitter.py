import json
import datetime
from typing import Dict, Any

class NdjsonEmitter:
    """Emitter producing NDJSON bytes for unified event schema"""
    def emit(self, event: Dict[str, Any]) -> bytes:
        # Build envelope with UTC timestamp
        out = {
            "type": event.get("type", ""),
            "session_id": event.get("session_id", ""),
            "data": event.get("data", {}),
            "ts": datetime.datetime.utcnow().isoformat(),
        }
        return (json.dumps(out) + "\n").encode("utf-8")