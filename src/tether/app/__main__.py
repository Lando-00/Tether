"""HTTP entry point — composition root for ``python -m tether_service.app``.

Loads ``.env`` (centralized here per _synthesis.md §4 Phase 2 step 26),
constructs typed Settings, builds the FastAPI app via ``create_app()``, and
hands it to uvicorn. Library users who construct ``Engine`` directly are
responsible for both ``load_dotenv()`` AND signal handling (LIBRARY mode);
see ``tether_service.__init__`` docstring.

Phase 3 follow-up: signal handling is now installed by the FastAPI
lifespan startup (see ``app/http/api.py::lifespan``) instead of here.
``uvicorn.run`` calls ``capture_signals`` which replaces any handlers
installed before it; the lifespan startup runs AFTER that, so our
handler wins. Without this fix, the force-exit timer never fired during
the run.
"""
import uvicorn
from dotenv import load_dotenv

from tether_service.app.http.api import create_app
from tether_service.config.settings import load_settings


def main():
    load_dotenv()

    settings = load_settings()
    app = create_app()
    uvicorn.run(app, host=settings.http.host, port=settings.http.port)


if __name__ == "__main__":
    main()
