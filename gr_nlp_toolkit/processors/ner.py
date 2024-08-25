import torch
from torch import nn

from transformers import AutoModel

from gr_nlp_toolkit.configs.ner_labels import ner_labels
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

from gr_nlp_toolkit.models.ner_model import NERBERTModel


model_params = {'dp': 0}


class NER(AbstractProcessor):
    """
    Named Entity Recognition (NER) processor class.

    This class performs NER using a pre-trained BERT model. It initializes the model,
    loads the necessary components, and provides functionality to process documents
    and perform NER on them.

    Attributes:
        I2L (list): A list of label names for the NER task.
        output_size (int): The number of output labels.
        _model (NERBERTModel): The NER model based on BERT.
        softmax (nn.Softmax): Softmax function for output normalization.
        device (torch.device): Device on which the model is loaded.
    """

    def __init__(self, model_path=None, device='cpu', entities=18,):
        """
        Initializes the NER class with the specified parameters.

        Args:
            model_path (str, optional): Path to the pre-trained model. Defaults to None.
            device (str, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
            entities (int, optional): Number of entity labels. Should be set to 18. Defaults to 18.

        Raises:
            ValueError: If the number of entities is not 18.
        """

        # Entities are the semantic catgories of the NER task (more info: http://nlp.cs.aueb.gr/theses/smyrnioudis_bsc_thesis.pdf)
        if entities == 18:
            self.I2L = ner_labels
            self.output_size = len(self.I2L)
        else:
            raise ValueError('Entities should be set to 18')

         # Initialize the BERT model
        bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self._model = NERBERTModel(bert_model, self.output_size, **model_params)
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device(device)
        self._model.to(self.device)
        self._model.eval()

        # load the pretrained model if provided
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)



    def __call__(self, doc: Document) -> Document:
        """
        Processes a document to perform Named Entity Recognition.

        Args:
            doc (Document): The document to process.

        Returns:
            Document: The document with NER tags assigned to the tokens.
        """
         
        # Get the input ids and text length of the document
        input_ids, text_len = next(iter(doc.dataloader))['input']
        
        # Perform NER with the model
        output = self._model(input_ids.to(self.device), text_len.to(self.device))
        predictions = self.softmax(output)
        predictions = torch.argmax(predictions[0], axis=-1).detach().cpu().numpy()

        # map predictions -> tokens, special tokens are not included
        i = 0
        for mask, pred in zip(doc.token_mask, predictions[1: len(predictions) - 1]):
            if mask:
                token = doc.tokens[i]
                token.ner = self.I2L[pred]
                i+=1
        
        return doc