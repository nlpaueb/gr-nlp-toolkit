import unittest

from gr_nlp_toolkit.domain.token import Token


class MyTestCase(unittest.TestCase):
    def test_new_token_object(self):
        token = Token(['α'])
        self.assertEqual(['α'], token.subwords)


if __name__ == '__main__':
    unittest.main()
