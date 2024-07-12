import unittest

from gr_nlp_toolkit.processors.g2g import G2G
from gr_nlp_toolkit.processors.g2g import detect_language
from gr_nlp_toolkit.domain.document import Document

class MyTestCase(unittest.TestCase):

    def test_g2g_lstm(self):

        g2g = G2G(mode="LSTM",model_path="gr_nlp_toolkit/RBNLM_weights/LSTM_LM_50000_char_120_32_512.pt", tokenizer_path="gr_nlp_toolkit/RBNLM_weights/RBNLMtextVectorizer.pkl")
        self.assertIsNotNone(g2g.model)
        self.assertIsNotNone(g2g.text_vectorizer)
        self.assertIsNotNone(g2g.LM)


        doc = Document("o volos kai h larisa einai poleis ths thessalias")
        doc = g2g(doc)
        self.assertEqual(detect_language(doc.text), 'greek')
        self.assertEqual(doc.text.split()[5], "είναι")

    
    def test_g2g_transformer(self):
            
        g2g = G2G(mode="transformer", model_path="AUEB-NLP/ByT5_g2g")
        self.assertIsNotNone(g2g.model)

        doc = Document('"o volos kai h larisa einai poleis ths thessalias"')
        doc = g2g(doc)
        self.assertEqual(detect_language(doc.text), 'greek')
        self.assertEqual(doc.text.split()[5], "είναι")

        
if __name__ == '__main__':
    unittest.main()
