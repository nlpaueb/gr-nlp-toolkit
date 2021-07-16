import unittest

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.pos import POS
from gr_nlp_toolkit.processors.tokenizer import Tokenizer
from gr_nlp_toolkit.I2Ls.pos_I2Ls import I2L_POS, properties_POS


class MyTestCase(unittest.TestCase):
    def test_pos_with_one_example(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))

        pos = POS()
        self.assertIsNotNone(pos.model)
        self.assertIsNotNone(pos.system)
        doc = pos(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.upos)
            self.assertIsNotNone(token.feats)
            self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))


if __name__ == '__main__':
    unittest.main()
