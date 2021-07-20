from typing import List

from gr_nlp_toolkit.data.downloader_gdrive import GDriveDownloader
from gr_nlp_toolkit.data.processor_cache import ProcessorCache
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
        self._processor_cache = ProcessorCache(GDriveDownloader())

        self._processors = []
        processors = set(processors.split(","))
        available_processors = ['ner', 'pos', 'dp']

        # Adding the tokenizer processor
        self._processors.append(Tokenizer())
        for p in processors:
            if p == available_processors[0]:
                ner_path = self._processor_cache.get_processor_path('ner')
                self._processors.append(NER(model_path=ner_path))
            elif p == available_processors[1]:
                pos_path = self._processor_cache.get_processor_path('pos')
                self._processors.append(POS(model_path=pos_path))
            elif p == available_processors[2]:
                dp_path = self._processor_cache.get_processor_path('dp')
                self._processors.append(DependencyParsing(model_path=dp_path))
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
