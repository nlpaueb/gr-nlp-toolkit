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

    def __init__(self, model_path=None, entities=18):
        # bert model init
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)

        if entities == 18:
            self.I2L = I2L_IOBES_18
            self.output_size = size_IOBES_18
        elif entities == 4:
            self.I2L = I2L_IOBES_4
            self.output_size = size_IOBES_4

        self.model = NERBERTModel(bert_model, self.output_size, **model_params)

        # system init
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.system = pw.System(self.model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        # TODO: when we download the model, we should uncomment to next line
        # self.system.load_model_state(model_path)

    def __call__(self, doc: Document) -> Document:
        predictions = self.system.predict(doc.dataloader, perform_last_activation=True)

        predictions = numpy.argmax(predictions['outputs'][0], axis=-1)

        if len(predictions) == len(doc.input_ids):
            for pred, token in zip(predictions, doc.tokens):
                token.ner = self.I2L[pred]

        return doc
