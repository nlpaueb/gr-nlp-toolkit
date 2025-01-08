import unittest

from gr_nlp_toolkit.configs.dp_labels import dp_labels
from gr_nlp_toolkit.configs.ner_labels import ner_labels
from gr_nlp_toolkit.configs.pos_labels import pos_labels, pos_properties
from gr_nlp_toolkit.pipeline.pipeline import Pipeline



class TestPipeline(unittest.TestCase):
    # The unit tests test the transformer-based g2g processor. If you want to test the LSTM-based processor, you can change 'g2g' to 'g2g_lite'
    def test_using_all_processors(self):
        nlp = Pipeline('dp,pos,ner,g2g') 

        sentences = ["Η Ιταλία κέρδισε την Αγγλία στον τελικό του Euro το 2021",
                     "Το ποιηματάκι το έγραψε ο διάσημος ποιητής, Νίκος Νικολαϊδης",
                     "Uparxoun autoi pou kerdizoun apo mia katastash kai autoi pou xanoun"]
        for sent in sentences:
            doc = nlp(sent)

            for token in doc.tokens:
                print(token.text, token.ner, token.upos, token.feats, token.head, token.deprel)
                self.assertIsNotNone(token.ner)
                self.assertTrue(token.ner in ner_labels)
                self.assertIsNotNone(token.head)
                self.assertIsNotNone(token.deprel)
                # We have to add plus one, because the cls token is removed
                self.assertTrue(token.head in range(0, len(doc.tokens) + 1))
                self.assertTrue(token.deprel in dp_labels)
                self.assertIsNotNone(token.upos)
                self.assertTrue(token.upos in pos_labels['upos'])

                self.assertIsNotNone(token.feats)
                self.assertEqual(len(list(token.feats.keys())), len(pos_properties[token.upos]))

                for feat, value in token.feats.items():
                    self.assertTrue(feat in pos_properties[token.upos])
                    self.assertTrue(value in pos_labels[feat])
                    print(token.text, token.ner, token.upos, token.feats, token.head, token.deprel)
                    self.assertIsNotNone(token.ner)
                    self.assertTrue(token.ner in ner_labels)
                    self.assertIsNotNone(token.head)
                    self.assertIsNotNone(token.deprel)
                    # We have to add plus one, because the cls token is removed
                    self.assertTrue(token.head in range(0, len(doc.tokens) + 1))
                    self.assertTrue(token.deprel in dp_labels)
                    self.assertIsNotNone(token.upos)
                    self.assertTrue(token.upos in pos_labels['upos'])

    def test_annotations_are_same_with_multiple_configurations(self):
        nlp = Pipeline('dp,pos,ner,g2g')
        doc = nlp("Uparxoun autoi pou kerdizoun apo mia katastash kai autoi pou xanoun")

        deprels_preds = []
        upos_preds = []
        ner_preds = []
        for token in doc.tokens:
            deprels_preds.append(token.deprel)
            upos_preds.append(token.upos)
            ner_preds.append(token.ner)

        nlp = Pipeline('dp,g2g')
        doc = nlp("Uparxoun autoi pou kerdizoun apo mia katastash kai autoi pou xanoun")
        new_deprels_preds = []

        for token in doc.tokens:
            new_deprels_preds.append(token.deprel)

        nlp = Pipeline('pos,g2g')
        doc = nlp("Uparxoun autoi pou kerdizoun apo mia katastash kai autoi pou xanoun")
        new_upos_preds =[]

        for token in doc.tokens:
            new_upos_preds.append(token.upos)

        nlp = Pipeline('ner,g2g')
        doc = nlp("Uparxoun autoi pou kerdizoun apo mia katastash kai autoi pou xanoun")
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
            self.assertTrue(token.ner in ner_labels)
            self.assertIsNone(token.head)
            self.assertIsNone(token.deprel)
            self.assertFalse(token.head in range(0, len(doc.tokens)))
            self.assertFalse(token.deprel in dp_labels)
            self.assertIsNone(token.upos)
            self.assertFalse(token.upos in pos_labels['upos'])

            for feat, value in token.feats.items():
                self.assertFalse(feat in pos_properties[token.upos])
                self.assertFalse(value in pos_labels[feat])


if __name__ == '__main__':
    unittest.main()
