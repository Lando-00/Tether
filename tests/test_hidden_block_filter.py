import unittest
from llm_service.protocol.api import HiddenBlockFilter

class TestHiddenBlockFilter(unittest.TestCase):
    def collect_visible(self, tokens, on_hidden=None):
        hf = HiddenBlockFilter(on_hidden_chunk=on_hidden)
        return [hf.feed(tok).to_ui_visible for tok in tokens]

    def test_hidden_blocks(self):
        cases = [
            (['Hello', ' ', 'world'], ['Hello', ' ', 'world']),
            (['Start', '<think>', 'secret', '</think>', 'End'], ['Start', '', '', '', 'End']),
            (['A', '<reflection>', 'r1', '</reflection>', 'B'], ['A', '', '', '', 'B']),
            (['X', '<!--', 'comment', '-->', 'Y'], ['X', '', '', '', 'Y']),
        ]
        for tokens, expected in cases:
            with self.subTest(tokens=tokens):
                self.assertEqual(self.collect_visible(tokens), expected)

    def test_on_hidden_chunk_callback(self):
        tokens = ['T1', '<think>', 'hidden', '</think>', 'T2']
        hidden_chunks = []
        def on_hidden(chunk):
            hidden_chunks.append(chunk)
        visible = self.collect_visible(tokens, on_hidden)
        self.assertEqual(visible, ['T1', '', '', '', 'T2'])
        self.assertTrue(any('hidden' in chunk for chunk in hidden_chunks))

if __name__ == '__main__':
    unittest.main()
