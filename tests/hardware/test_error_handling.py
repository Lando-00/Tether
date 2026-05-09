"""
Hardware test: threading.excepthook is available and catches exceptions.
Converted from root test_error_handling.py diagnostic script.
Requires Python 3.8+ (always true for this project).
Run with: pytest -m hardware tests/hardware/test_error_handling.py
"""
import sys
import threading
import time

import pytest


@pytest.mark.hardware
def test_python_version_supports_thread_excepthook():
    """Python version must be 3.8+ for threading.excepthook support."""
    assert sys.version_info >= (3, 8), (
        f"Python {sys.version} is too old — threading.excepthook requires 3.8+"
    )


@pytest.mark.hardware
def test_thread_exception_handler_catches_exceptions():
    """threading.excepthook can intercept exceptions raised in daemon threads.

    Preserves the assertion intent of the original test_error_handling.py:
    caught_exceptions must be non-empty after the daemon thread raises.
    """
    caught_exceptions = []

    def custom_handler(args):
        caught_exceptions.append(args.exc_value)

    def failing_thread():
        time.sleep(0.1)
        raise RuntimeError("Test exception from thread")

    old_handler = threading.excepthook
    threading.excepthook = custom_handler
    try:
        t = threading.Thread(target=failing_thread, daemon=True)
        t.start()
        t.join()
        time.sleep(0.2)
    finally:
        threading.excepthook = old_handler

    assert len(caught_exceptions) > 0, (
        "threading.excepthook did not catch the exception — "
        "TVM/OpenCL errors from the MLC daemon thread will go undetected."
    )
