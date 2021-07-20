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


class DependencyParsing(AbstractProcessor):
    """
    DP class that takes a document and returns a document with tokens' head and deprels fields set
    """

    def __init__(self, model_path=None, device='cpu'):
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)

        self.I2L = I2L_deprels
        self.output_size = len(self.I2L)

        self._model = DependencyParsingModel(bert_model, self.I2L, 0)

        self.system = pw.System(self._model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        # TODO: we should uncomment the next line
        # self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict heads
        output_heads = 'heads'
        predictions_heads = self.system.predict(doc.dataloader, perform_last_activation=True,
                                                model_output_key=output_heads)
        predictions_heads = numpy.argmax(predictions_heads['outputs'][0], axis=-1)

        # predict deprels
        output_deprels = 'gathered_deprels'
        predictions_deprels = self.system.predict(doc.dataloader, perform_last_activation=True,
                                                model_output_key=output_deprels)
        predictions_deprels = numpy.argmax(predictions_deprels['outputs'][0], axis=-1)

        # map predictions -> tokens, special tokens are not included
        if len(predictions_heads[1: len(predictions_heads) - 1]) == len(doc.tokens) \
                and len(predictions_deprels[1: len(predictions_deprels) - 1]) == len(doc.tokens):
            for pred_head, pred_deprel, token in zip(predictions_heads[1: len(predictions_heads) - 1],
                                        predictions_deprels[1: len(predictions_deprels) - 1], doc.tokens):
                token.head = pred_head
                token.deprel = self.I2L[pred_deprel]
        return doc
