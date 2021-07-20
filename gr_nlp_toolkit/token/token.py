from typing import List


class Token:
    """
    Token class which represents a word/token
    """

    def __init__(self, subwords: List[str]):
        """
            Create a Token object setting possible parameters other than the text as None
            @:param: text: The text of the token
        """

        # the text
        self.text = ""

        # the subwords
        self.subwords = subwords

        # the ids
        self._ids = []

        # Named Entity Recognition parameters
        # the named entity
        self.ner = None

        # Part of Speech Tagging parameters
        # the universal pos tag
        self.upos = None
        # the universal morphological features
        self.feats = {}

        # Dependency Parsing parameters
        # the dependant word index in the sentence
        self.head = None
        # the label of the relation between the specific word and the dependant one
        self.deprel = None

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def subwords(self):
        return self._subwords

    @subwords.setter
    def subwords(self, value):
        self._subwords = value

    @property
    def ner(self):
        return self._ner

    @ner.setter
    def ner(self, value):
        self._ner = value

    @property
    def upos(self):
        return self._upos

    @upos.setter
    def upos(self, value):
        self._upos = value

    @property
    def feats(self):
        return self._feats

    @feats.setter
    def feats(self, value):
        self._feats = value

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = value

    @property
    def deprel(self):
        return self._deprel

    @deprel.setter
    def deprel(self, value):
        self._deprel = value
