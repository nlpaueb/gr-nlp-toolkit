from typing import List

from gr_nlp_toolkit.document.document import Document


class Pipeline:
    """
    The central class of the toolkit. A pipeline is created after a list of processors are specified. The user can
    then annotate a document by using the __call__ method of the Pipeline
    """
    def __init__(self, processors: List[str]):
        self._processors = processors

    def __call__(self, text: str):
        """
        Annotate a text
        :param text: A string containing the text to be annotated
        :return: A Document object containing the annotations
        """
        # Create a document from the text
        self._doc = Document(text)
