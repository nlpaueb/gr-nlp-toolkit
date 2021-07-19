import unittest

from gr_nlp_toolkit.I2Ls.dp_I2Ls import I2L_deprels
from gr_nlp_toolkit.I2Ls.ner_I2Ls import I2L_IOBES_18
from gr_nlp_toolkit.I2Ls.pos_I2Ls import I2L_POS, properties_POS
from gr_nlp_toolkit.pipeline.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def test_using_all_processors(self):
        nlp = Pipeline('dp,pos,ner')
        doc = nlp("Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021")

        for token in doc.tokens:
            self.assertIsNotNone(token.ner)
            self.assertTrue(token.ner in I2L_IOBES_18)
            self.assertIsNotNone(token.head)
            self.assertIsNotNone(token.deprel)
            self.assertTrue(token.head in range(0, len(doc.tokens)))
            self.assertTrue(token.deprel in I2L_deprels)
            self.assertIsNotNone(token.upos)
            self.assertTrue(token.upos in I2L_POS['upos'])

            self.assertIsNotNone(token.feats)
            self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))

            for feat, value in token.feats.items():
                self.assertTrue(feat in properties_POS[token.upos])
                self.assertTrue(value in I2L_POS[feat])

    def test_using_only_one_processor(self):
        nlp = Pipeline('ner')
        doc = nlp("Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021")

        for token in doc.tokens:
            self.assertIsNotNone(token.ner)
            self.assertTrue(token.ner in I2L_IOBES_18)
            self.assertIsNone(token.head)
            self.assertIsNone(token.deprel)
            self.assertFalse(token.head in range(0, len(doc.tokens)))
            self.assertFalse(token.deprel in I2L_deprels)
            self.assertIsNone(token.upos)
            self.assertFalse(token.upos in I2L_POS['upos'])


            for feat, value in token.feats.items():
                self.assertFalse(feat in properties_POS[token.upos])
                self.assertFalse(value in I2L_POS[feat])




if __name__ == '__main__':
    unittest.main()
