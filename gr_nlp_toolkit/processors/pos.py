import numpy
import torch
from torch import nn

import pytorch_wrapper as pw

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor
from gr_nlp_toolkit.I2Ls.pos_I2Ls import I2L_POS, properties_POS

from transformers import AutoModel

from gr_nlp_toolkit.processors.pos_model import POSModel

pretrained_bert_name = 'nlpaueb/bert-base-greek-uncased-v1'


class POS(AbstractProcessor):
    """
    POS class that takes a document and returns a document with tokens' upos and feats fields set
    """

    def __init__(self, model_path=None):
        # bert model init
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)

        self.properties_POS = properties_POS
        self.feat_to_I2L = I2L_POS
        self.feat_to_size = {k: len(v) for k, v in self.feat_to_I2L.items()}

        self.model = POSModel(bert_model, self.feat_to_size, 0)

        # system init
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.system = pw.System(self.model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        if model_path != None:
            with open(model_path, 'rb') as f:
                self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict
        predictions = {}
        for feat in self.feat_to_I2L.keys():
            predictions[feat] = numpy.argmax(self.system.predict(doc.dataloader, perform_last_activation=True,
                                                                 model_output_key=feat, verbose=False)['outputs'][0],
                                             axis=-1)

        # set upos
        upos_predictions = predictions['upos']
        if len(upos_predictions[1: len(upos_predictions) - 1]) == len(doc.tokens):
            for pred, token in zip(upos_predictions[1: len(upos_predictions) - 1], doc.tokens):
                token.upos = self.feat_to_I2L['upos'][pred]

        # set features
        for feat in self.feat_to_I2L.keys():
            if feat != 'upos':
                current_predictions = predictions[feat]
                if len(current_predictions[1: len(current_predictions) - 1]) == len(doc.tokens):
                    for pred, token in zip(current_predictions[1: len(current_predictions) - 1], doc.tokens):
                        if feat in self.properties_POS[token.upos]:
                            token.feats[feat] = self.feat_to_I2L[feat][pred]

        return doc
