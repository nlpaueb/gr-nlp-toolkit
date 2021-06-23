import torch
from torch import nn

import pytorch_wrapper as pw

from gr_nlp_toolkit.document.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

from transformers import AutoModel

from gr_nlp_toolkit.processors.ner_model import NERBERTModel

pretrained_bert_name = 'nlpaueb/bert-base-greek-uncased-v1'
model_path = ""
model_output_size = None
model_params = None

model_params = {'dp': 0}


class NER(AbstractProcessor):
    def __init__(self):
        # bert model init
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)
        self.model = NERBERTModel(bert_model, model_output_size, **model_params)

        # system init
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.system = pw.System(self.model, last_activation=nn.Softmax(dim=-1), device=torch.device(device))

        # load the pretrained model
        self.system.load_model_state(model_path)

    """
    NER class that takes a document and returns a document with ner fields set
    """

    def __call__(self, doc: Document) -> Document:
        predictions = self.system.predict(doc.dataloader)

        if len(predictions) == doc.tokens:
            print("ok")
            for pred, token in zip(predictions, doc.tokens):
                token.ner = pred

        return doc
