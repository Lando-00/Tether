"""Tether app package marker.

The FastAPI app is constructed explicitly via
``tether_service.app.http.api.create_app()`` — this module no longer
pre-builds an ``app`` instance at import time. Per _synthesis.md §4
Phase 2 step 23 (library-first; no import-time side effects).
"""
