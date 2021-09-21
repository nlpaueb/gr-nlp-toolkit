from abc import ABC, abstractmethod

from gr_nlp_toolkit.domain.document import Document


class AbstractProcessor(ABC):
    @abstractmethod
    def __call__(self, doc : Document):
        pass