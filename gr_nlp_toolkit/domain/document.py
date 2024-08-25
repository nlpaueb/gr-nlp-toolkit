

class Document:
    """
    Document class that represents an annotated text
    """

    def __init__(self, text: str):
        """
        Create a Document object setting possible parameters other than the text as None

        Keyword arguments:
        param text: The text of the document
        """
        self._text = text

        self._input_ids = None
        self._token_mask = None

        self._tokens = None

        self._dataloader = None

        self._subword2word = None


    @property
    def text(self):
        """
        Return the original text of the document
        """
        return self._text

    @text.setter
    def text(self, value):
        self._text = value


    @property
    def tokens(self):
        """
        A list of Tokens containing the tokens of the text as well as token level annotations
        """
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value


    @property
    def input_ids(self):
        """
         A tensor of shape [1,mseq] containing the input ids created with the BERT tokenizer
        """
        return self._input_ids

    @input_ids.setter
    def input_ids(self, value):
        self._input_ids = value


    @property
    def token_mask(self):
        """
        A tensor of shape [1,mseq] containign zeros at the positions of the input_ids tensor that map to subword tokens that are non first subword tokens
        """
        return self._token_mask

    @token_mask.setter
    def token_mask(self, value):
        self._token_mask = value


    @property
    def dataloader(self):
        return self._dataloader

    
    @dataloader.setter
    def dataloader(self, value):
        self._dataloader = value

    @property
    def subword2word(self):
        """
        A mapping for each subword to the word
        """
        return self._subword2word

    @subword2word.setter
    def subword2word(self, value):
        self._subword2word = value