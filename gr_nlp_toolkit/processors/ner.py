import numpy
import torch
from torch import nn

import pytorch_wrapper as pw
from transformers import AutoModel

from gr_nlp_toolkit.I2Ls.ner_I2Ls import I2L_IOBES_18, I2L_IOBES_4
from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor


from gr_nlp_toolkit.processors.ner_model import NERBERTModel


model_params = {'dp': 0}


class NER(AbstractProcessor):
    """
    NER class that takes a document and returns a document with ner fields set
    """
    def __init__(self, model_path=None, device='cpu', entities=18,):

        if entities == 18:
            self.I2L = I2L_IOBES_18
            self.output_size = len(self.I2L)
        else:
            raise ValueError('Entities should be set to 18')

        # model init
        bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self._model = NERBERTModel(bert_model, self.output_size, **model_params)

        self.system = pw.System(self._model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        if model_path != None:
            with open(model_path, 'rb') as f:
                self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict
        predictions = self.system.predict(doc.dataloader, perform_last_activation=True, verbose=False)
        predictions = numpy.argmax(predictions['outputs'][0], axis=-1)

        # map predictions -> tokens, special tokens are not included
        i = 0
        for mask, pred in zip(doc.token_mask, predictions[1: len(predictions) - 1]):
            if mask:
                token = doc.tokens[i]
                token.ner = self.I2L[pred]
                i+=1

        return doc
