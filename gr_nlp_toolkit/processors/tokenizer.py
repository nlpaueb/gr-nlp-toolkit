from typing import List, Tuple

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

import unicodedata

from gr_nlp_toolkit.token.token import Token

from transformers import AutoTokenizer

tokenizer_greek = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')


def strip_accents_and_lowercase(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()


def create_ids(text: str) -> List[int]:
    # encode text
    return tokenizer_greek.encode(text)


def convert_to_tokens(input_ids: List[int]) -> List[str]:
    return tokenizer_greek.convert_ids_to_tokens(input_ids, skip_special_tokens=True)


def create_mask_and_tokens(input_tokens: List[int]) -> Tuple[List[str], List[Token]]:
    mask = []
    tokens = []
    # for each token
    for t in input_tokens:
        # it isn't a sub-word
        if not t.startswith("##"):
            #  create a token object
            tokenObj = Token([t])
            tokens.append(tokenObj)
            mask.append('0')
        else:
            # add sub-words to token
            tokenObj.subwords.append(t)
            mask.append('1')
    return mask, tokens


class Tokenizer(AbstractProcessor):
    """
    Tokenizer class that takes a document as an input with the text field set, tokenizes and returns a document with
    all fields set
    """
    def __call__(self, doc: Document) -> Document:
        # get document's text and strip accent and lowercase
        doc.text = strip_accents_and_lowercase(doc.text)
        # create ids
        doc.input_ids = create_ids(doc.text)
        # create mask and tokens
        doc.mask, doc.tokens = create_mask_and_tokens(convert_to_tokens(doc.input_ids))
        return doc
