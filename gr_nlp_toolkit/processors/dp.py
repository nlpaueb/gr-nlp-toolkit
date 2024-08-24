import torch
from torch import nn

from transformers import AutoModel

from gr_nlp_toolkit.configs.dp_labels import dp_labels
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

from gr_nlp_toolkit.models.dp_model import DPModel


class DP(AbstractProcessor):
    """
    Dependency Parsing (DP) processor class.

    This class performs dependency parsing using a pre-trained BERT model. It initializes the model,
    loads the necessary components, and provides functionality to process documents
    and assign head and dependency relation (deprel) tags to tokens.

     Attributes:
        I2L (list): A list of dependency relation labels.
        output_size (int): The number of output labels.
        _model (DPModel): The dependency parsing model based on BERT.
        softmax (nn.Softmax): Softmax function for output normalization.
        device (torch.device): Device on which the model is loaded.
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Initializes the DP class with the specified parameters.

        Args:
            model_path (str, optional): Path to the pre-trained model. Defaults to None.
            device (str, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """

        self.I2L = dp_labels
        self.output_size = len(self.I2L)

         # Initialize the BERT model
        bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self._model = DPModel(bert_model, self.I2L, 0)

        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device(device)
        self._model.to(self.device)
        self._model.eval()

        # Load the pretrained model if provided
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)

    def __call__(self, doc: Document) -> Document:
        """
        Processes a document to perform dependency parsing.

        Args:
            doc (Document): The document to process.

        Returns:
            Document: The document with head and deprel tags assigned to the tokens.
        """

        # Predict heads
        input_ids, text_len = next(iter(doc.dataloader))['input']

        output_heads = 'heads'
       
        predictions_heads = self._model(input_ids.to(self.device), text_len.to(self.device))
        predictions_heads = self.softmax(predictions_heads[output_heads])
        predictions_heads = torch.argmax(predictions_heads[0], axis=-1).detach().cpu().numpy()

        # Predict dependency relations (deprels)
        output_deprels = 'gathered_deprels'
        
        predictions_deprels = self._model(input_ids.to(self.device), text_len.to(self.device))
        predictions_deprels = self.softmax(predictions_deprels[output_deprels])
        predictions_deprels = torch.argmax(predictions_deprels[0], axis=-1).detach().cpu().numpy()

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
