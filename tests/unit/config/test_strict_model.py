"""Tests for StrictModel (M6) — extra=forbid + frozen behavior."""
import pytest
from pydantic import Field, ValidationError

from tether_service.config._strict import StrictModel


class _Inner(StrictModel):
    name: str = "x"


class _Outer(StrictModel):
    field: str
    inner: _Inner = Field(default_factory=_Inner)


def test_strict_model_extra_forbid():
    with pytest.raises(ValidationError):
        _Outer(field="ok", unknown="boom")  # type: ignore[call-arg]


def test_strict_model_frozen():
    obj = _Outer(field="ok")
    with pytest.raises(ValidationError):
        obj.field = "changed"  # type: ignore[misc]


def test_strict_model_default_factory():
    obj = _Outer(field="ok")
    assert isinstance(obj.inner, _Inner)
    assert obj.inner.name == "x"
