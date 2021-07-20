import unittest

from transformers import AutoModel

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.ner import NER
from gr_nlp_toolkit.processors.tokenizer import Tokenizer

from gr_nlp_toolkit.I2Ls.ner_I2Ls import I2L_IOBES_18


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

    def test_ner_with_one_example(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))

        ner = NER(MyTestCase.bert_model)

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

        ner = NER(MyTestCase.bert_model)
        self.assertIsNotNone(ner._model)
        self.assertIsNotNone(ner.system)
        doc = ner(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.ner)
            self.assertTrue(token.ner in I2L_IOBES_18)

    def test_ner_with_value_exception(self):
        with self.assertRaises(ValueError):
            NER(MyTestCase.bert_model, entities=2)


if __name__ == '__main__':
    unittest.main()
