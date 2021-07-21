import unittest

from transformers import AutoModel

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.ner import NER
from gr_nlp_toolkit.processors.tokenizer import Tokenizer

from gr_nlp_toolkit.I2Ls.ner_I2Ls import I2L_IOBES_18


class MyTestCase(unittest.TestCase):

    def test_ner_with_one_example(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))

        ner = NER(entities=18)

        self.assertEqual(69, ner.output_size)
        self.assertIsNotNone(ner._model)
        self.assertIsNotNone(ner.system)
        doc = ner(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.ner)
            self.assertTrue(token.ner in I2L_IOBES_18)

    def test_ner_with_one_example_with_subwords(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('ενα ποιηματακι'))

        ner = NER()
        self.assertIsNotNone(ner._model)
        self.assertIsNotNone(ner.system)
        doc = ner(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.ner)
            self.assertTrue(token.ner in I2L_IOBES_18)

    def test_ner_with_value_exception(self):
        with self.assertRaises(ValueError):
            NER(entities=2)


if __name__ == '__main__':
    unittest.main()
