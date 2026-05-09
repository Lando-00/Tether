"""HTTP entry point — composition root for ``python -m tether_service.app``.

Loads ``.env`` (centralized here per _synthesis.md §4 Phase 2 step 26),
constructs typed Settings, builds the FastAPI app via ``create_app()``, and
hands it to uvicorn. Library users who construct ``Engine`` directly are
responsible for calling ``load_dotenv()`` themselves if they want .env
support (see ``tether_service.__init__`` docstring).
"""
import uvicorn
from dotenv import load_dotenv

from tether_service.app.http.api import create_app
from tether_service.config.settings import load_settings


def main():
    # _synthesis.md §4 Phase 2 step 26: centralize .env loading. Tests load
    # .env via conftest.py; library users load it themselves; only this
    # composition root touches it for the HTTP entry point.
    load_dotenv()
    settings = load_settings()
    app = create_app()
    uvicorn.run(app, host=settings.http.host, port=settings.http.port)


if __name__ == "__main__":
    main()
