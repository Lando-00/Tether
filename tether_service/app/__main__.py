"""HTTP entry point — composition root for ``python -m tether_service.app``.

Loads ``.env`` (centralized here per _synthesis.md §4 Phase 2 step 26),
installs the SERVER-mode :class:`SignalSupervisor` (Phase 3 step 31),
constructs typed Settings, builds the FastAPI app via ``create_app()``, and
hands it to uvicorn. Library users who construct ``Engine`` directly are
responsible for both ``load_dotenv()`` AND signal handling (LIBRARY mode);
see ``tether_service.__init__`` docstring.
"""
import uvicorn
from dotenv import load_dotenv

from tether_service.app.http.api import create_app
from tether_service.config.settings import load_settings
from tether_service.runtime.signal_supervisor import SignalSupervisor


def main():
    load_dotenv()

    SignalSupervisor(max_shutdown_sec=5.0).install()

    settings = load_settings()
    app = create_app()
    uvicorn.run(app, host=settings.http.host, port=settings.http.port)


if __name__ == "__main__":
    main()
