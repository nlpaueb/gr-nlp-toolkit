from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.processors.dp import DP
from gr_nlp_toolkit.processors.ner import NER
from gr_nlp_toolkit.processors.pos import POS
from gr_nlp_toolkit.processors.g2g import G2G

from gr_nlp_toolkit.processors.tokenizer import Tokenizer
from huggingface_hub import hf_hub_download

from typing import Literal
import torch

from transformers import logging
logging.set_verbosity_error()

def get_device_name() -> Literal["mps", "cuda", "cpu"]:
    """
    Returns the name of the device where this module is running.

    This is a simple implementation that doesn't cover cases when more powerful GPUs are available 
    and not a primary device ('cuda:0') or MPS device is available but not configured properly:
    https://pytorch.org/docs/master/notes/mps.html

    Returns:
        Literal["mps", "cuda", "cpu"]: Device name, like 'cuda' or 'cpu'.

    Examples:
        >>> torch.cuda.is_available = lambda: True
        >>> torch.backends.mps.is_available = lambda: False
        >>> get_device_name()
        'cuda'

        >>> torch.cuda.is_available = lambda: False
        >>> torch.backends.mps.is_available = lambda: True
        >>> get_device_name()
        'mps'

        >>> torch.cuda.is_available = lambda: False
        >>> torch.backends.mps.is_available = lambda: False
        >>> get_device_name()
        'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Pipeline:
    """
    The central class of the toolkit. A pipeline is created after a list of processors are specified. The user can
    then annotate a document by using the __call__ method of the Pipeline
    
    Attributes:
        _processors: A list of the processors that will be used in the pipeline
        _processor_cache: A ProcessorCache object that is used to download the processors
        device: The device where the pipeline will run

    """

    def __init__(self, processors: str, use_cpu: bool = False):
        """ 
        Initializes the pipeline with the specified processors

        Args:
            processors: A string with the names of the processors you want to load, available values: 'ner', 'por', 'dp
            use_cpu: A boolean that specifies if the pipeline will run on the CPU
        """

        # if the user wants to use the CPU, we set the device to 'cpu'
        if(use_cpu):
            self.device = "cpu"
        else:
            self.device = get_device_name()

        self._processors = []

        processors = set(processors.split(","))

        # ner: Named Entity Recognition Processor 
        # pos: Part of Speech Recognition Processor
        # dp: Dependency Parsing 
        # g2g: Greeklish to Greek Transliteration Processor (ByT5 model)
        # g2g_lite: Greeklish to Greek Transliteration Processor (LSTM model)
        available_processors = ['ner', 'pos', 'dp', 'g2g_lite', 'g2g']


        # Adding the g2g processor, which must be the first in the pipeline
        if("g2g_lite" in processors):
            self._processors.append(G2G(mode="LSTM", model_path="gr_nlp_toolkit/RBNLM_weights/LSTM_LM_50000_char_120_32_512.pt", tokenizer_path="gr_nlp_toolkit/RBNLM_weights/RBNLMtextVectorizer.pkl", device=self.device))
            processors.remove("g2g_lite")
        elif("g2g" in processors):
            self._processors.append(G2G(mode="transformer", model_path="AUEB-NLP/ByT5_g2g", device=self.device))
            processors.remove("g2g")

            
        # Adding the tokenizer processor
        self._processors.append(Tokenizer())
        for p in processors:
            if p == available_processors[0]:
                ner_path = hf_hub_download(repo_id="AUEB-NLP/gr-nlp-toolkit", filename="ner_processor")
                self._processors.append(NER(model_path=ner_path, device=self.device))
            elif p == available_processors[1]:
                pos_path = hf_hub_download(repo_id="AUEB-NLP/gr-nlp-toolkit", filename="pos_processor")
                self._processors.append(POS(model_path=pos_path, device=self.device))
            elif p == available_processors[2]:
                dp_path = hf_hub_download(repo_id="AUEB-NLP/gr-nlp-toolkit", filename="dp_processor")
                self._processors.append(DP(model_path=dp_path, device=self.device))
            else:
                raise Exception(f"Invalid processor name, please choose one of {available_processors}")

    def __call__(self, text: str) -> Document:
        
        """
        Annotate a text with the processors present in the pipeline

        Args:
            text: The text that will be annotated
        """

        # Create a document from the text
        self._doc = Document(text)

        # Pass the document through every processor
        for processor in self._processors:
            # print(processor)
            processor(self._doc)

        return self._doc
    
if __name__ == "__main__": 


    nlp = Pipeline("g2g,ner,dp,pos")
   
    txts = ["Uparxoun autoi pou kerdizoun apo mia katastash kai autoi pou hanoun",
            "o volos kai h larisa einai poleis ths thessalias",
            "Η Αθήνα είναι η μεγαλύτερη πόλη της Ελλάδας"]

    for txt in txts:

        doc = nlp(txt)
        
        print(doc.text)
        for token in doc.tokens:
            print(f"{token.text}: {token.ner}, {token.upos}, {token.feats}, {token.head}, {token.deprel}") # the text of the token
            

