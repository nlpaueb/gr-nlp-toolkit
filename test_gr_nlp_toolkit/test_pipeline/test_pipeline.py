import unittest

from gr_nlp_toolkit.I2Ls.dp_I2Ls import I2L_deprels
from gr_nlp_toolkit.I2Ls.ner_I2Ls import I2L_IOBES_18
from gr_nlp_toolkit.I2Ls.pos_I2Ls import I2L_POS, properties_POS
from gr_nlp_toolkit.pipeline.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    def test_using_all_processors(self):
        nlp = Pipeline('dp,pos,ner')

        sentences = ["Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021",
                     "Το ποιηματάκι το έγραψε ο διάσημος ποιητής, Νίκος Νικολαϊδης"]
        for sent in sentences:
            doc = nlp(sent)

            for token in doc.tokens:
                print(token.text, token.ner, token.upos, token.feats, token.head, token.deprel)
                self.assertIsNotNone(token.ner)
                self.assertTrue(token.ner in I2L_IOBES_18)
                self.assertIsNotNone(token.head)
                self.assertIsNotNone(token.deprel)
                # We have to add plus one, because the cls token is removed
                self.assertTrue(token.head in range(0, len(doc.tokens) + 1))
                self.assertTrue(token.deprel in I2L_deprels)
                self.assertIsNotNone(token.upos)
                self.assertTrue(token.upos in I2L_POS['upos'])

                self.assertIsNotNone(token.feats)
                self.assertEqual(len(list(token.feats.keys())), len(properties_POS[token.upos]))

                for feat, value in token.feats.items():
                    self.assertTrue(feat in properties_POS[token.upos])
                    self.assertTrue(value in I2L_POS[feat])
                    print(token.text, token.ner, token.upos, token.feats, token.head, token.deprel)
                    self.assertIsNotNone(token.ner)
                    self.assertTrue(token.ner in I2L_IOBES_18)
                    self.assertIsNotNone(token.head)
                    self.assertIsNotNone(token.deprel)
                    # We have to add plus one, because the cls token is removed
                    self.assertTrue(token.head in range(0, len(doc.tokens) + 1))
                    self.assertTrue(token.deprel in I2L_deprels)
                    self.assertIsNotNone(token.upos)
                    self.assertTrue(token.upos in I2L_POS['upos'])

    def test_annotations_are_same_with_multiple_configurations(self):
        nlp = Pipeline('dp,pos,ner')
        doc = nlp("Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021")

        deprels_preds = []
        upos_preds = []
        ner_preds = []
        for token in doc.tokens:
            deprels_preds.append(token.deprel)
            upos_preds.append(token.upos)
            ner_preds.append(token.ner)

        nlp = Pipeline('dp')
        doc = nlp("Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021")
        new_deprels_preds = []

        for token in doc.tokens:
            new_deprels_preds.append(token.deprel)

        nlp = Pipeline('pos')
        doc = nlp("Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021")
        new_upos_preds =[]

        for token in doc.tokens:
            new_upos_preds.append(token.upos)

        nlp = Pipeline('ner')
        doc = nlp("Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021")
        new_ner_preds =[]
        for token in doc.tokens:
            new_ner_preds.append(token.ner)

        self.assertEqual(new_deprels_preds, deprels_preds)
        self.assertEqual(new_upos_preds, upos_preds)
        self.assertEqual(new_ner_preds, ner_preds)



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
