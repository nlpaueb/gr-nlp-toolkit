import unittest

from transformers import AutoModel

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.dp import DP
from gr_nlp_toolkit.processors.tokenizer import Tokenizer

from gr_nlp_toolkit.I2Ls.dp_I2Ls import I2L_deprels


class MyTestCase(unittest.TestCase):

    def test_dp_with_one_example(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))

        dp = DP()
        self.assertIsNotNone(dp._model)
        self.assertIsNotNone(dp.system)
        doc = dp(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.head)
            self.assertIsNotNone(token.deprel)
            self.assertTrue(token.head in range(0, len(tokens)))
            self.assertTrue(token.deprel in I2L_deprels)

    def test_dp_with_one_example_with_subwords(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('ενα ποιηματακι'))

        # bert model init
        dp = DP()

        self.assertIsNotNone(dp._model)
        self.assertIsNotNone(dp.system)
        doc = dp(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.head)
            self.assertIsNotNone(token.deprel)
            self.assertTrue(token.head in range(0, len(tokens)))
            self.assertTrue(token.deprel in I2L_deprels)
