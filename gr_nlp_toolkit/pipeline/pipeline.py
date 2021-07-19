from typing import List

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.dp import DependencyParsing
from gr_nlp_toolkit.processors.ner import NER
from gr_nlp_toolkit.processors.pos import POS
from gr_nlp_toolkit.processors.tokenizer import Tokenizer


class Pipeline:
    """
    The central class of the toolkit. A pipeline is created after a list of processors are specified. The user can
    then annotate a document by using the __call__ method of the Pipeline
    """

    def __init__(self, processors: str):
        self._processors = []
        processors = set(processors.split(","))
        available_processors = ['ner', 'pos', 'dp']

        # Adding the tokenizer processor
        self._processors.append(Tokenizer())
        for p in processors:
            if p == available_processors[0]:
                self._processors.append(NER())
            elif p == available_processors[1]:
                self._processors.append(POS())
            elif p == available_processors[2]:
                self._processors.append(DependencyParsing())
            else:
                raise Exception(f"Invalid processor name, please choose one of {available_processors}")

    def __call__(self, text: str) -> Document:
        """
        Annotate a text
        :param text: A string containing the text to be annotated
        :return: A Document object containing the annotations
        """
        # Create a document from the text
        self._doc = Document(text)

        # Pass the document through every processor
        for processor in self._processors:
            processor(self._doc)

        return self._doc
