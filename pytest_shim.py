# pytest.py
"""
Minimal pytest shim to allow tests importing pytest to run under unittest.
"""
import unittest

def skip(msg=None, *args, **kwargs):
    """Skip test, mapping pytest.skip to unittest.SkipTest."""
    reason = msg or ""
    raise unittest.SkipTest(reason)

# decorator for fixtures
def fixture(*_args, **_kwargs):
    def decorator(f):
        return f
    return decorator

# mark.integration decorator stub
class mark:
    @staticmethod
    def integration(f):
        return f
