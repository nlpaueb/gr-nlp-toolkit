import unittest

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
        mask, tokens, subword2word = create_mask_and_tokens(tokens, [247, 6981])

        self.assertEqual(2, len(mask))
        self.assertEqual([True, True], mask)
        self.assertEqual(2, len(tokens))
        self.assertEqual(1, len(tokens[0].subwords))
        self.assertEqual(1, len(tokens[1].subwords))
        self.assertEqual(247, tokens[0]._ids[0])
        self.assertEqual(6981, tokens[1]._ids[0])
        self.assertEqual('ο', tokens[0].subwords[0])
        self.assertEqual('ποιητης', tokens[1].subwords[0])
        self.assertEqual('ο', tokens[0].text)
        self.assertEqual('ποιητης', tokens[1].text)
        self.assertEqual(len(subword2word.keys()), 3)
        self.assertEqual(subword2word[1], 1)
        self.assertEqual(subword2word[2], 2)

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
        mask, tokens, subword2word = create_mask_and_tokens(tokens, [370, 6623, 701])

        self.assertEqual(3, len(mask))
        self.assertEqual([True, True, False], mask)
        self.assertEqual(2, len(tokens))
        self.assertEqual(1, len(tokens[0].subwords))
        self.assertEqual(2, len(tokens[1].subwords))
        self.assertEqual(370, tokens[0]._ids[0])
        self.assertEqual(6623, tokens[1]._ids[0])
        self.assertEqual(701, tokens[1]._ids[1])
        self.assertEqual('ενα', tokens[0].subwords[0])
        self.assertEqual('ποιηματα', tokens[1].subwords[0])
        self.assertEqual('##κι', tokens[1].subwords[1])
        self.assertEqual('ενα', tokens[0].text)
        self.assertEqual('ποιηματακι', tokens[1].text)
        self.assertEqual(len(subword2word.keys()), 4)
        self.assertEqual(subword2word[1], 1)
        self.assertEqual(subword2word[2], 2)
        self.assertEqual(subword2word[3], 2)

    def test_tokenizer(self):
        tokenizer = Tokenizer()
        doc = tokenizer(Document('Ο ποιητής'))
        # document has all field set
        self.assertIsNotNone(doc.text)
        self.assertIsNotNone(doc.input_ids)
        self.assertIsNotNone(doc.token_mask)
        self.assertIsNotNone(doc.tokens)
        self.assertIsNotNone(doc.subword2word)

    def test_create_dataset_and_dataloader(self):
        input_ids = [101, 370, 6623, 701, 102]
        dataset, dataloader = create_dataset_and_dataloader(input_ids)
        self.assertIsNotNone(dataset.input_ids)
        self.assertIsNotNone(dataloader.dataset)
        self.assertEqual(dataset, dataloader.dataset)
        self.assertEqual(dataset.input_ids, [input_ids])


if __name__ == '__main__':
    unittest.main()
