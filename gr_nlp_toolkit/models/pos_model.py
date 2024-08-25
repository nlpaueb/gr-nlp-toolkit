from torch import nn
from gr_nlp_toolkit.models.util import create_mask_from_length

class POSModel(nn.Module):
    """
    Part-Of-Speech (POS) tagging model class based on BERT.

    This class defines a POS model using a pre-trained BERT model 
    with a dropout and multiple linear layers on top.

    Attributes:
        _bert_model (AutoModel): The pre-trained BERT model.
        _dp (nn.Dropout): Dropout layer for regularization.
        _linear_dict (nn.ModuleDict): Dictionary of linear layers for different features.
    """

    def __init__(self, bert_model, feat_to_size, dp):
        """
        Initializes the POSModel with the specified parameters.

        Args:
            bert_model (AutoModel): The pre-trained BERT model.
            feat_to_size (dict): A dictionary mapping feature names to the size of their output layers.
            dp (float): Dropout probability.
        """

        super(POSModel, self).__init__()
        self._bert_model = bert_model
        self._dp = nn.Dropout(dp)

        self._linear_dict = nn.ModuleDict({feat: nn.Linear(768, feat_to_size[feat]) for feat in feat_to_size})

    def forward(self, text, text_len):
        """
        Performs a forward pass of the model.

        Args:
            text (torch.Tensor): Input tensor containing token IDs.
            text_len (torch.Tensor): Tensor containing the lengths of each sequence in the batch.

        Returns:
            dict: A dictionary containing the output tensors for each feature.
        """
        
        attention_mask = create_mask_from_length(text_len, text.shape[1])
        bert_output = self._dp(self._bert_model(text, attention_mask=attention_mask)[0])

        output_dict = {feat: self._linear_dict[feat](bert_output) for feat in self._linear_dict}
        return output_dict

