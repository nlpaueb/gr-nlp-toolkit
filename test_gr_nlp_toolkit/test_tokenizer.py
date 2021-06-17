import unittest

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.tokenizer import *


class TestTokenizer(unittest.TestCase):
    def test_strip_accents_and_lowercase1(self):
        result = strip_accents_and_lowercase('ποιητής')
        self.assertEqual('ποιητης', result)

    def test_strip_accents_and_lowercase2(self):
        result = strip_accents_and_lowercase('ΠΟΙΗΤΗΣ')
        self.assertEqual('ποιητης', result)

    """"
        Tests with no sub-words:
    """

    def test_create_ids_without_subwords(self):
        ids = create_ids('ο ποιητης')
        # 2 special tokens + 2 given words
        self.assertEqual(4, len(ids))

    def test_create_tokens_without_subwords(self):
        ids = [101, 247, 6981, 102]
        tokens = convert_to_tokens(ids)
        # 2 words, special tokens are not included
        self.assertEqual(2, len(tokens))
        self.assertEqual('ο', tokens[0])
        self.assertEqual('ποιητης', tokens[1])

    def test_create_mask_and_tokens_without_subwords(self):
        tokens = ['ο', 'ποιητης']
        mask, tokens = create_mask_and_tokens(tokens)

        self.assertEqual(2, len(mask))
        self.assertEqual(['0', '0'], mask)
        self.assertEqual(2, len(tokens))
        self.assertEqual(1, len(tokens[0].subwords))
        self.assertEqual(1, len(tokens[1].subwords))
        self.assertEqual('ο', tokens[0].subwords[0])
        self.assertEqual('ποιητης', tokens[1].subwords[0])

    """"
        Tests with sub-words:
    """

    def test_create_ids_with_subwords(self):
        ids = create_ids('ενα ποιηματακι')
        # 2 special tokens + 1 word without sub-words + 1 word with 1 sub-word
        self.assertEqual(5, len(ids))

    def test_create_tokens_with_subwords(self):
        ids = [101, 370, 6623, 701, 102]
        tokens = convert_to_tokens(ids)
        # 2 words, special tokens are not included
        self.assertEqual(3, len(tokens))
        self.assertEqual('ενα', tokens[0])
        self.assertEqual('ποιηματα', tokens[1])
        self.assertEqual('##κι', tokens[2])

    def test_create_mask_and_tokens_with_subwords(self):
        tokens = ['ενα', 'ποιηματα', '##κι']
        mask, tokens = create_mask_and_tokens(tokens)
        self.assertEqual(3, len(mask))
        self.assertEqual(['0', '0', '1'], mask)
        self.assertEqual(2, len(tokens))
        self.assertEqual(1, len(tokens[0].subwords))
        self.assertEqual(2, len(tokens[1].subwords))
        self.assertEqual('ενα', tokens[0].subwords[0])
        self.assertEqual('ποιηματα', tokens[1].subwords[0])
        self.assertEqual('##κι', tokens[1].subwords[1])

    def test_tokenizer(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))
        # document has all field set
        self.assertIsNotNone(doc.text)
        self.assertIsNotNone(doc.input_ids)
        self.assertIsNotNone(doc.mask)
        self.assertIsNotNone(doc.tokens)


if __name__ == '__main__':
    unittest.main()
