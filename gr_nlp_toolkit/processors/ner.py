import numpy
import torch
from torch import nn

import pytorch_wrapper as pw

from gr_nlp_toolkit.I2Ls.ner_I2Ls import I2L_IOBES_18, size_IOBES_18, I2L_IOBES_4, size_IOBES_4
from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

from transformers import AutoModel

from gr_nlp_toolkit.processors.ner_model import NERBERTModel

pretrained_bert_name = 'nlpaueb/bert-base-greek-uncased-v1'
model_params = {'dp': 0}


class NER(AbstractProcessor):
    """
    NER class that takes a document and returns a document with ner fields set
    """

    def __init__(self, model_path=None, device='cpu', entities=18,):
        # bert model init
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)

        if entities == 18:
            self.I2L = I2L_IOBES_18
            self.output_size = size_IOBES_18
        elif entities == 4:
            self.I2L = I2L_IOBES_4
            self.output_size = size_IOBES_4
        else:
            raise ValueError('Entities should be set to 18 or 4')

        self._model = NERBERTModel(bert_model, self.output_size, **model_params)

        self.system = pw.System(self._model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        if model_path != None:
            with open(model_path, 'rb') as f:
                self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict
        predictions = self.system.predict(doc.dataloader, perform_last_activation=True)
        predictions = numpy.argmax(predictions['outputs'][0], axis=-1)

        # map predictions -> tokens, special tokens are not included
        if len(predictions[1: len(predictions) - 1]) == len(doc.tokens):
            for pred, token in zip(predictions[1: len(predictions) - 1], doc.tokens):
                token.ner = self.I2L[pred]

        return doc
