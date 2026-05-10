"""Tether — local LLM inference with function calling.

Public API:
    from tether import Engine, Settings, load_settings

Importing this package MUST NOT pull in FastAPI, MLC, or Brave. Every
concrete provider/parser/store/tool is lazy-imported inside the consumer
(e.g., ``Engine.from_settings``). Per _synthesis.md §4 Phase 2 step 22
(library-first import surface).

.env loading:
    Library users who construct ``Engine`` directly are responsible for
    calling ``dotenv.load_dotenv()`` themselves if they want .env support
    (e.g. for ``BRAVE_API_KEY``). The HTTP entry point
    (``python -m tether.app``) calls it on their behalf, and the
    test suite loads it via ``conftest.py``. Per _synthesis.md §4 Phase 2
    step 26 (centralized .env loading).
"""
from tether.config import Settings, load_settings
from tether.engine import Engine

__all__ = ["Engine", "Settings", "load_settings"]
