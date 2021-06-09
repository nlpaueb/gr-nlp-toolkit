from unittest import TestCase
from gr_nlp_toolkit.process.sentence_process import SentenceProcess


class TestSplitSentence(TestCase):

    def test_always_passes(self):
        sentence_splitter = SentenceProcess("hello my name is Mary")
        result = sentence_splitter.split_sentence()
        self.assertEqual(5, len(result))
