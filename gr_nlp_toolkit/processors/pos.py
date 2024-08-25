import torch
from torch import nn

from transformers import AutoModel

from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor
from gr_nlp_toolkit.configs.pos_labels import pos_labels, pos_properties


from gr_nlp_toolkit.models.pos_model import POSModel


class POS(AbstractProcessor):
    """
    Part-Of-Speech (POS) processor class.

    This class performs POS tagging using a pre-trained BERT model. It initializes the model,
    loads the necessary components, and provides functionality to process documents
    and assign POS tags and features to tokens.

     Attributes:
        properties_POS (dict): Dictionary containing properties for POS tags.
        feat_to_I2L (dict): Dictionary mapping feature names to label lists.
        feat_to_size (dict): Dictionary mapping feature names to the size of their label lists.
        _model (POSModel): The POS model based on BERT.
        softmax (nn.Softmax): Softmax function for output normalization.
        device (torch.device): Device on which the model is loaded.
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Initializes the POS class with the specified parameters.

        Args:
            model_path (str, optional): Path to the pre-trained model. Defaults to None.
            device (str, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """

        self.properties_POS = pos_properties
        self.feat_to_I2L = pos_labels
        self.feat_to_size = {k: len(v) for k, v in self.feat_to_I2L.items()}

        # model init
        bert_model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
        self._model = POSModel(bert_model, self.feat_to_size, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device(device)
        self._model.to(self.device)
        self._model.eval()

        # load the pretrained model
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)

    def __call__(self, doc: Document) -> Document:
        """
        Processes a document to perform Part-Of-Speech tagging and assign features.

        Args:
            doc (Document): The document to process.

        Returns:
            Document: The document with POS tags and features assigned to the tokens.
        """

        predictions = {}

        input_ids, text_len = next(iter(doc.dataloader))['input']

        for feat in self.feat_to_I2L.keys():
            output = self._model(input_ids.to(self.device), text_len.to(self.device))
            output = self.softmax(output[feat])

            
            predictions[feat] = torch.argmax(output[0], axis=-1).detach().cpu().numpy()

        # set upos
        upos_predictions = predictions['upos']
        i = 0
        for mask, pred in zip(doc.token_mask, upos_predictions[1: len(upos_predictions) - 1]):
            if mask:
                token = doc.tokens[i]
                token.upos = self.feat_to_I2L['upos'][pred]
                # Advance to the next word (not subtoken)
                i+=1

        # set features
        for feat in self.feat_to_I2L.keys():
            if feat != 'upos':
                current_predictions = predictions[feat]
                i = 0
                for mask, pred in zip(doc.token_mask, current_predictions[1: len(current_predictions) - 1]):
                    if mask:
                        token = doc.tokens[i]
                        if feat in self.properties_POS[token.upos]:
                            token.feats[feat] = self.feat_to_I2L[feat][pred]
                        # Advance to the next word (not subtoken)
                        i += 1

        return doc
