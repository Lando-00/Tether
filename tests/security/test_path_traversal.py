"""
Security tests: path-traversal prevention in MLCProvider.

§security R-pathtraversal (§4 Phase 0A):
Any model_name that could traverse outside dist_root must raise ValueError
BEFORE any filesystem access occurs.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from tether_service.providers.mlc.provider import MLCProvider


def _make_provider(tmp_path: Path) -> MLCProvider:
    """Return an MLCProvider rooted at tmp_path (no real GPU needed)."""
    # We deliberately do NOT create any dist layout — the validation must
    # fire before any filesystem touch.
    provider = object.__new__(MLCProvider)
    provider.dist_root = tmp_path
    provider.libs_dir = tmp_path / "libs"
    provider.device = "auto"
    provider.max_tokens = 1024
    import threading
    provider._cache_lock = threading.Lock()
    provider._engine_cache = {}
    provider._init_locks = {}
    return provider


TRAVERSAL_NAMES = [
    "../../etc/passwd",
    "..\\windows\\system32",
    "/etc/passwd",
    "C:\\Windows\\System32",
    "../sibling",
    "a/b",
    "a\\b",
    ".",
    "..",
    "",
    "model\x00name",             # null byte
    "a" * 129,                   # too long (>128 chars)
    "model name",                 # space
    "model:name",                 # colon
    "model!name",                 # exclamation
]

VALID_NAMES = [
    "Qwen3-4B-q4f16_0-MLC",
    "Qwen2.5-7B-q4f16_0-MLC",
    "my-model",
    "model123",
    "a" * 128,                    # exactly 128 chars — boundary
]


@pytest.mark.parametrize("bad_name", TRAVERSAL_NAMES)
def test_validate_model_name_rejects_traversal(tmp_path, bad_name):
    """Malicious model_name values must raise ValueError before any fs access."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError, match="invalid model_name"):
        provider._validate_model_name(bad_name)


@pytest.mark.parametrize("good_name", VALID_NAMES)
def test_validate_model_name_accepts_valid(tmp_path, good_name):
    """Clean model names that resolve within dist_root must pass validation."""
    provider = _make_provider(tmp_path)
    # Should not raise; no return value
    provider._validate_model_name(good_name)


@pytest.mark.parametrize("bad_name", ["../../etc/passwd", "../sibling", "/etc/passwd"])
def test_get_context_window_rejects_traversal(tmp_path, bad_name):
    """get_context_window must validate model_name before touching filesystem."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError, match="invalid model_name"):
        provider.get_context_window(bad_name)


@pytest.mark.parametrize("bad_name", ["../../etc/passwd", "../sibling", "/etc/passwd"])
def test_get_engine_rejects_traversal(tmp_path, bad_name):
    """_get_engine must validate model_name before touching filesystem."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError, match="invalid model_name"):
        provider._get_engine(bad_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_name", ["../../etc/passwd", "../sibling", "/etc/passwd"])
async def test_ensure_engine_rejects_traversal(tmp_path, bad_name):
    """_ensure_engine must validate model_name before touching filesystem."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError, match="invalid model_name"):
        await provider._ensure_engine(bad_name)


def test_absolute_path_rejected(tmp_path):
    """Absolute paths must always be rejected regardless of dist_root location."""
    provider = _make_provider(tmp_path)
    abs_path = str(tmp_path)  # Even a path that IS dist_root — it contains a separator
    with pytest.raises(ValueError, match="invalid model_name"):
        provider._validate_model_name(abs_path)
