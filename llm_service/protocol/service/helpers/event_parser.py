import json
class EventParser:
    def parse(self, raw: bytes) -> dict:
        text = raw.decode("utf-8").strip()
        return json.loads(text)