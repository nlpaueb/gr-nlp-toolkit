import numpy
import torch
from torch import nn

import pytorch_wrapper as pw
from transformers import AutoModel

from gr_nlp_toolkit.I2Ls.dp_I2Ls import I2L_deprels
from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

from gr_nlp_toolkit.processors.dp_model import DPModel


class DP(AbstractProcessor):
    """
    DP class that takes a document and returns a document with tokens' head and deprels fields set
    """

    def __init__(self, model_path=None, device='cpu'):

        self.I2L = I2L_deprels
        self.output_size = len(self.I2L)

        # model init
        bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self._model = DPModel(bert_model, self.I2L, 0)

        self.system = pw.System(self._model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        if model_path != None:
            with open(model_path, 'rb') as f:
                self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict heads
        output_heads = 'heads'
        predictions_heads = self.system.predict(doc.dataloader, perform_last_activation=True,
                                                model_output_key=output_heads, verbose=False)
        predictions_heads = numpy.argmax(predictions_heads['outputs'][0], axis=-1)

        # predict deprels
        output_deprels = 'gathered_deprels'
        predictions_deprels = self.system.predict(doc.dataloader, perform_last_activation=True,
                                                  model_output_key=output_deprels, verbose=False)
        predictions_deprels = numpy.argmax(predictions_deprels['outputs'][0], axis=-1)

        # map predictions -> tokens, special tokens are not included
        i = 0
        for mask, pred_head, pred_deprel in zip(doc.token_mask, predictions_heads[1: len(predictions_heads) - 1],
                                                       predictions_deprels[1: len(predictions_deprels) - 1]):
            if mask:
                token = doc.tokens[i]
                token.head = doc.subword2word[pred_head]
                token.deprel = self.I2L[pred_deprel]
                i +=1

        return doc
