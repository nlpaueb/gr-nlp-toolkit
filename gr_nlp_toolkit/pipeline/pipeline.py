from os.path import expanduser

from transformers import AutoModel

from gr_nlp_toolkit.data.downloader_gdrive import GDriveDownloader
from gr_nlp_toolkit.data.processor_cache import ProcessorCache
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.processors.dp import DP
from gr_nlp_toolkit.processors.ner import NER
from gr_nlp_toolkit.processors.pos import POS
from gr_nlp_toolkit.processors.g2g import G2G

from gr_nlp_toolkit.processors.tokenizer import Tokenizer

import os
import warnings
# warnings.filterwarnings("ignore")

class Pipeline:
    """
    The central class of the toolkit. A pipeline is created after a list of processors are specified. The user can
    then annotate a document by using the __call__ method of the Pipeline
    """

    def __init__(self, processors: str):
        """ Load the processors

           Keyword arguments:
            processors: A list with the names of the processors you want to load, available values: 'ner', 'por', 'dp'
        """

        home = expanduser("~")
        sep = os.sep
        cache_path = home + sep + ".cache" + sep + "gr_nlp_toolkit"
        self._processor_cache = ProcessorCache(GDriveDownloader() , cache_path)

        self._processors = []
        processors = set(processors.split(","))
        available_processors = ['ner', 'pos', 'dp', 'g2g_lstm', 'g2g_transformer']


        # Adding the g2g processor, which must be the first in the pipeline

        if("g2g_lstm" in processors):
            self._processors.append(G2G(mode="LSTM", model_path="gr_nlp_toolkit/tmp/LSTM_LM_50000_char_120_32_512.pt", tokenizer_path="gr_nlp_toolkit/tmp/RBNLMtextVectorizer.pkl"))
            processors.remove("g2g_lstm")
        elif("g2g_transformer" in processors):
            self._processors.append(G2G(mode="transformer", model_path="gr_nlp_toolkit/tmp/ByT5-TV"))
            processors.remove("g2g_transformer")

            
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
                self._processors.append(DP(model_path=dp_path))
            else:
                raise Exception(f"Invalid processor name, please choose one of {available_processors}")

    def __call__(self, text: str) -> Document:
        
        """
        Annotate a text

        Keyword arguments:
        param text: A string or a list of strings containing the text to be annotated
        return: A Document object containing the annotations
        """

        # Create a document from the text
        self._doc = Document(text)

        # Pass the document through every processor
        for processor in self._processors:
            # print(processor)
            processor(self._doc)

        return self._doc
    
if __name__ == "__main__": 

    nlp = Pipeline("g2g_lstm,ner")

    doc = nlp("o volos kai h larisa einai poleis ths thessalias")
    # doc = nlp("ο Βολος και η Λαρισα ειναι πολεις της θεσσαλίας")
    print(doc.text)
    for token in doc.tokens:
        print(token.text) # the text of the token
        
        print(token.ner) # the named entity label in IOBES encoding : str
        
        print(token.upos) # the UPOS tag of the token
        print(token.feats) # the morphological features for the token
        
        print(token.head) # the head of the token
        print(token.deprel) # the dependency relation between the current token and its head

