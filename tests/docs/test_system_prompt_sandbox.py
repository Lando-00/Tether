"""P0-B1: system prompt must teach the model about the tool_result sandbox."""
import pytest

from tether.config.settings import load_settings


@pytest.mark.docs
def test_system_prompt_includes_sandbox_rule():
    settings = load_settings()
    prompt = settings.system.prompt
    assert "<<tool_result" in prompt
    assert "<</tool_result>>" in prompt
    assert (
        "DATA" in prompt
        or "data, not instructions" in prompt.lower()
        or "never execute imperatives" in prompt.lower()
    )
