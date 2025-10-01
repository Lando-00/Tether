import unittest
from llm_service.protocol.api import PrefixedToolCallDetector

class FakeLogger:
    def __init__(self):
        self.messages = []
    def debug(self, msg, *args):
        self.messages.append(("debug", msg % args))
    def warning(self, msg, *args):
        self.messages.append(("warning", msg % args))
    def info(self, msg, *args):
        self.messages.append(("info", msg % args))
    def error(self, msg, *args):
        self.messages.append(("error", msg % args))
    def exception(self, msg, *args):
        self.messages.append(("exception", msg % args))

class TestToolCallDetector(unittest.TestCase):
    def test_simple_prose(self):
        detector = PrefixedToolCallDetector(logger=FakeLogger())
        status, payload = detector.feed("Hello")
        # A non-tool token should be classified as prose with the raw content
        self.assertEqual(status, "prose")
        self.assertEqual(payload, "Hello")

    def test_complete_tool_call(self):
        detector = PrefixedToolCallDetector(logger=FakeLogger())
        # simulate start of call
        status, name = detector.feed("__tool_test(")
        self.assertEqual(status, "call_started")
        self.assertEqual(name, "test")
        # simulate end of call
        status2, args = detector.feed(")")
        self.assertEqual(status2, "call_complete")
        self.assertEqual(args, "")

    def test_incremental_tool_call(self):
        detector = PrefixedToolCallDetector(logger=FakeLogger())
        # simulate partial tool call tokens
        status1, name1 = detector.feed("__tool_inc(")
        self.assertEqual(status1, "call_started")
        self.assertEqual(name1, "inc")
        # mid-stream tokens should be undecided
        status2, payload2 = detector.feed("param1,")
        self.assertEqual(status2, "undecided")
        self.assertIsNone(payload2)
        status3, payload3 = detector.feed("param2)")
        # should complete call with raw args 'param1,param2'
        self.assertEqual(status3, "call_complete")
        self.assertEqual(payload3, "param1,param2")

    def test_newline_rejection(self):
        detector = PrefixedToolCallDetector(logger=FakeLogger())
        # feed first token, then a newline
        status1, payload1 = detector.feed("abc")
        self.assertEqual(status1, "undecided")
        self.assertIsNone(payload1)
        # feeding newline should force prose with full raw
        status2, payload2 = detector.feed("\n")
        self.assertEqual(status2, "prose")
        self.assertEqual(payload2, "abc\n")

if __name__ == '__main__':
    unittest.main()
