from abc import ABC, abstractmethod

from gr_nlp_toolkit.document.document import Document


class AbstractProcessor(ABC):
    @abstractmethod
    def __call__(self, doc : Document):
        pass