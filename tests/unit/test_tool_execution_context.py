"""Tests for :class:`tether_service.core.types.ToolExecutionContext`.

Synthesis §4 Phase 4 step 41a; connector spec §4 footer.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tether_service.core.types import ToolExecutionContext


class TestToolExecutionContext:
    def test_tec_required_fields(self):
        """``session_id`` and ``turn_id`` are sufficient; the other fields
        default safely (last_user_message=None, user_confirmed_send=False)."""
        ctx = ToolExecutionContext(session_id="s", turn_id="t")
        assert ctx.session_id == "s"
        assert ctx.turn_id == "t"
        assert ctx.last_user_message is None
        assert ctx.user_confirmed_send is False

    def test_tec_all_fields(self):
        """All four fields can be set explicitly."""
        ctx = ToolExecutionContext(
            session_id="sid-1",
            turn_id="turn-abc",
            last_user_message="please send the message",
            user_confirmed_send=True,
        )
        assert ctx.session_id == "sid-1"
        assert ctx.turn_id == "turn-abc"
        assert ctx.last_user_message == "please send the message"
        assert ctx.user_confirmed_send is True

    def test_tec_frozen(self):
        """Frozen dataclass — assignment must raise FrozenInstanceError."""
        ctx = ToolExecutionContext(session_id="s", turn_id="t")
        with pytest.raises(FrozenInstanceError):
            ctx.session_id = "other"  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            ctx.user_confirmed_send = True  # type: ignore[misc]

    def test_tec_default_user_confirmed_send_false(self):
        """The default for ``user_confirmed_send`` MUST be False.

        Synthesis §10.8 #4: the regex classifier that may flip it to True
        ships in the WhatsApp/Gmail connector sessions; in this refactor
        it is unconditionally False.
        """
        ctx = ToolExecutionContext(session_id="s", turn_id="t")
        assert ctx.user_confirmed_send is False

    def test_tec_imports_from_core_types(self):
        """The canonical import path is ``tether_service.core.types``."""
        from tether_service.core.types import ToolExecutionContext as TEC

        assert TEC is ToolExecutionContext

    def test_tec_equality_by_value(self):
        """Frozen dataclasses compare by value (sanity check)."""
        a = ToolExecutionContext(session_id="s", turn_id="t")
        b = ToolExecutionContext(session_id="s", turn_id="t")
        assert a == b
        assert a is not b
