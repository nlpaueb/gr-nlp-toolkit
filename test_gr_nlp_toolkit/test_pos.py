import unittest

from transformers import AutoModel

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.pos import POS
from gr_nlp_toolkit.processors.tokenizer import Tokenizer
from gr_nlp_toolkit.I2Ls.pos_I2Ls import I2L_POS, properties_POS


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')


    def test_pos_with_one_example(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))

        pos = POS()
        self.assertIsNotNone(pos._model)
        self.assertIsNotNone(pos.system)
        doc = pos(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.upos)
            self.assertTrue(token.upos in I2L_POS['upos'])

            self.assertIsNotNone(token.feats)
            self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))

            for feat, value in token.feats.items():
                self.assertTrue(feat in properties_POS[token.upos])
                self.assertTrue(value in I2L_POS[feat])

    def test_pos_with_one_example_with_subwords(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('ενα ποιηματακι'))

        pos = POS(MyTestCase.bert_model)
        self.assertIsNotNone(pos._model)
        self.assertIsNotNone(pos.system)
        doc = pos(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.upos)
            self.assertTrue(token.upos in I2L_POS['upos'])

            self.assertIsNotNone(token.feats)
            self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))

            for feat, value in token.feats.items():
                self.assertTrue(feat in properties_POS[token.upos])
                self.assertTrue(value in I2L_POS[feat])

    def test_pos_with_one_example_with_subwords(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('ενα ποιηματακι'))

        pos = POS(MyTestCase.bert_model)
        self.assertIsNotNone(pos._model)
        self.assertIsNotNone(pos.system)
        doc = pos(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.upos)
            self.assertTrue(token.upos in I2L_POS['upos'])

            self.assertIsNotNone(token.feats)
            self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))

            for feat, value in token.feats.items():
                self.assertTrue(feat in properties_POS[token.upos])
                self.assertTrue(value in I2L_POS[feat])

    def test_pos_with_one_example_with_subwords(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('ενα ποιηματακι'))

        pos = POS()
        self.assertIsNotNone(pos._model)
        self.assertIsNotNone(pos.system)
        doc = pos(doc)

        tokens = doc.tokens
        for token in tokens:
            self.assertIsNotNone(token.upos)
            self.assertTrue(token.upos in I2L_POS['upos'])

            self.assertIsNotNone(token.feats)
            self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))

            for feat, value in token.feats.items():
                self.assertTrue(feat in properties_POS[token.upos])
                self.assertTrue(value in I2L_POS[feat])


if __name__ == '__main__':
    unittest.main()
