# import pytorch_wrapper.functional as pwF

from torch import nn
from gr_nlp_toolkit.models.util import create_mask_from_length


class NERBERTModel(nn.Module):
    """
    Named Entity Recognition (NER) model class based on BERT.

    This class defines a NER model using a pre-trained BERT model 
    with a dropout and a linear layer on top.

    Attributes:
        _bert_model (AutoModel): The pre-trained BERT model.
        _dp (nn.Dropout): Dropout layer for regularization.
        _output_linear (nn.Linear): Linear layer to produce model outputs.
    """
    def __init__(self, bert_model, model_output_size, dp):
        """
        Initializes the NERBERTModel with the specified parameters.

        Args:
            bert_model (AutoModel): The pre-trained BERT model.
            model_output_size (int): The size of the output layer.
            dp (float): Dropout probability.
        """
         
        super(NERBERTModel, self).__init__()
        self._bert_model = bert_model
        self._dp = nn.Dropout(dp)
        self._output_linear = nn.Linear(768, model_output_size)

    def forward(self, text, text_len):
        """
        Performs a forward pass of the model.

        Args:
            text (torch.Tensor): Input tensor containing token IDs.
            text_len (torch.Tensor): Tensor containing the lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The output of the linear layer after applying dropout and BERT.
        """

        # Create attention mask
        attention_mask = create_mask_from_length(text_len, text.shape[1])

        return self._output_linear(self._dp(self._bert_model(text, attention_mask=attention_mask)[0]))
    
