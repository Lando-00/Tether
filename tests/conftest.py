"""Root test configuration.

Provides global fixtures and warning filters that apply to the entire test suite.
"""
import warnings
import pytest


@pytest.fixture(autouse=True)
def _suppress_yoyo_warnings():
    """Suppress DeprecationWarnings from yoyo-migrations internals.

    yoyo 8.2.x uses datetime.utcnow() internally (Python 3.12 raises a
    DeprecationWarning). This fixture ensures -W error::DeprecationWarning
    sweeps stay clean without hiding warnings from project code.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"yoyo\.",
        )
        yield
