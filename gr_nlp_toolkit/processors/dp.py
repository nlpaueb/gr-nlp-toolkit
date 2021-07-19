import numpy
import torch
from torch import nn

import pytorch_wrapper as pw

from gr_nlp_toolkit.I2Ls.dp_I2Ls import I2L_deprels
from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

from transformers import AutoModel

from gr_nlp_toolkit.processors.dp_model import DependencyParsingModel

pretrained_bert_name = 'nlpaueb/bert-base-greek-uncased-v1'
dp = 0


class DependencyParsing(AbstractProcessor):
    """
    NER class that takes a document and returns a document with ner fields set
    """

    def __init__(self, model_path=None):
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)

        self.I2L_deprels = I2L_deprels
        self.output_size = len(self.I2L_deprels)

        self.model = DependencyParsingModel(bert_model, self.I2L_deprels, dp)

        # system init
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.system = pw.System(self.model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        # TODO: we should uncomment the next line
        # self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict
        predictions = self.system.predict(doc.dataloader, perform_last_activation=True)
        predictions = numpy.argmax(predictions['outputs'][0], axis=-1)
        print(predictions)
        # map predictions -> tokens, special tokens are not included
        # if len(predictions[1: len(predictions) - 1]) == len(doc.tokens):
        #     for pred, token in zip(predictions[1: len(predictions) - 1], doc.tokens):
        #         token.ner = self.I2L[pred]

        return doc
