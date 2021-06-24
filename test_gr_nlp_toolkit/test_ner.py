import unittest

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.ner import NER
from gr_nlp_toolkit.processors.tokenizer import Tokenizer


class MyTestCase(unittest.TestCase):
    def test_ner(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))

        ner = NER()
        self.assertIsNotNone(ner.model)
        self.assertIsNotNone(ner.system)
        doc = ner(doc)



if __name__ == '__main__':
    unittest.main()
