from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

import unicodedata


class Tokenizer(AbstractProcessor):
    def __call__(self, doc: Document):
        pass

    def strip_accents_and_lowercase(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn').lower()
