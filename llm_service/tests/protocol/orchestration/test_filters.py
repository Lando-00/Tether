import pytest
from unittest.mock import MagicMock
from llm_service.protocol.orchestration.filters import HiddenBlockFilter
from llm_service.protocol.core.types import RoutedChunk

def test_visible_tokens():
    """Tests that regular tokens are passed through as visible."""
    filt = HiddenBlockFilter()
    tokens = ["This", " is", " a", " test."]
    for token in tokens:
        result = filt.feed(token)
        assert result == RoutedChunk(to_detector=token, to_ui_visible=token, to_ui_hidden="")

@pytest.mark.parametrize("start_tag, end_tag", [
    ("<think>", "</think>"),
    ("<reflection>", "</reflection>"),
    ("<!--", "-->"),
    ('<think attribute="value">', '</think>'),
])
def test_hidden_blocks(start_tag, end_tag):
    """Tests that content within various hidden blocks is correctly filtered."""
    filt = HiddenBlockFilter(expose_hidden_thoughts=True)
    stream = ["Visible", " ", start_tag, " some", " hidden", " text ", end_tag, " more", " visible"]
    
    # Before the block
    res1 = filt.feed(stream[0])
    assert res1 == RoutedChunk(to_detector="Visible", to_ui_visible="Visible", to_ui_hidden="")
    res2 = filt.feed(stream[1])
    assert res2 == RoutedChunk(to_detector=" ", to_ui_visible=" ", to_ui_hidden="")

    # The hidden block itself
    for token in stream[2:7]:
        result = filt.feed(token)
        assert result.to_detector == ""
        assert result.to_ui_visible == ""
        assert result.to_ui_hidden == token

    # After the block
    res8 = filt.feed(stream[7])
    assert res8 == RoutedChunk(to_detector=" more", to_ui_visible=" more", to_ui_hidden="")
    res9 = filt.feed(stream[8])
    assert res9 == RoutedChunk(to_detector=" visible", to_ui_visible=" visible", to_ui_hidden="")

def test_expose_hidden_false():
    """Tests that hidden thoughts are not exposed when the flag is False."""
    filt = HiddenBlockFilter(expose_hidden_thoughts=False)
    stream = ["<think>", "hidden", "</think>"]
    for token in stream:
        result = filt.feed(token)
        assert result.to_ui_hidden == ""

def test_on_hidden_callback():
    """Tests that the on_hidden_chunk callback is called for hidden tokens."""
    mock_callback = MagicMock()
    filt = HiddenBlockFilter(on_hidden_chunk=mock_callback)
    
    stream = ["Visible", "<think>", "hidden1", "hidden2", "</think>"]
    for token in stream:
        filt.feed(token)
        
    mock_callback.assert_any_call("<think>")
    mock_callback.assert_any_call("hidden1")
    mock_callback.assert_any_call("hidden2")
    mock_callback.assert_any_call("</think>")
    assert mock_callback.call_count == 4

def test_filter_handles_split_tokens_correctly():
    """
    Tests how the filter behaves when tags are split across multiple tokens.
    The current implementation processes token by token, so it won't detect a split tag.
    This test confirms the current behavior.
    """
    filt = HiddenBlockFilter()
    
    # The start tag is split
    res1 = filt.feed("<th")
    assert res1.to_ui_hidden == ""
    assert res1.to_ui_visible == "<th"
    
    res2 = filt.feed("ink>")
    assert res2.to_ui_hidden == ""
    assert res2.to_ui_visible == "ink>"

    # A token that looks like an end tag but isn't, because we are not in a hidden block
    res3 = filt.feed("</think>")
    assert res3.to_ui_hidden == ""
    assert res3.to_ui_visible == "</think>"

def test_multiple_blocks_in_stream():
    """Tests that the filter can handle multiple hidden blocks in a single stream."""
    filt = HiddenBlockFilter()
    stream = [
        "Visible1",
        "<think>", "Hidden1", "</think>",
        "Visible2",
        "<!--", "Comment", "-->",
        "Visible3"
    ]
    
    # Visible1
    assert filt.feed(stream[0]).to_ui_visible == "Visible1"
    
    # <think> block
    assert filt.feed(stream[1]).to_ui_hidden == "<think>"
    assert filt.feed(stream[2]).to_ui_hidden == "Hidden1"
    assert filt.feed(stream[3]).to_ui_hidden == "</think>"
    
    # Visible2
    assert filt.feed(stream[4]).to_ui_visible == "Visible2"
    
    # <!-- block
    assert filt.feed(stream[5]).to_ui_hidden == "<!--"
    assert filt.feed(stream[6]).to_ui_hidden == "Comment"
    assert filt.feed(stream[7]).to_ui_hidden == "-->"
    
    # Visible3
    assert filt.feed(stream[8]).to_ui_visible == "Visible3"
