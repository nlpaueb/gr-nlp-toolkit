class Document:
    """
    Document class that represents an annotated text
    """
    def __init__(self, text: str):
        """
        Create a Document object setting possible parameters other than the text as None
        :param text: The text of the document
        """
        self._text = text

        self._input_ids = None
        self._token_mask = None

        self._tokens = None

    """
    Return the original text of the document
    """
    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    """
    A list of Tokens containing the tokens of the text as well as token level annotations
    """
    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    """
    # A tensor of shape [1,mseq] containing the input ids created with the BERT tokenizer
    """
    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, value):
        self._input_ids = value

    """
    A tensor of shape [1,mseq] containign zeros at the positions of the input_ids tensor that map to
    subword tokens that are non first subword tokens
    """
    @property
    def token_mask(self):
        return self._token_mask

    @token_mask.setter
    def token_mask(self, value):
        self._token_mask = value