import pytorch_wrapper.functional as pwF

from torch import nn


class POSModel(nn.Module):

    def __init__(self, bert_model, feat_to_size, dp):
        """
        :param bert_model:  The bert model nn.Module
        :param dp: the drop out probability
        :param feat_to_size: a dict mapping from a string feature to an int ( the number of outputs for the feature)
        """
        super(POSModel, self).__init__()
        self._bert_model = bert_model
        self._dp = nn.Dropout(dp)

        self._linear_dict = nn.ModuleDict({feat: nn.Linear(768, feat_to_size[feat]) for feat in feat_to_size})

    def forward(self, text, text_len):
        attention_mask = pwF.create_mask_from_length(text_len, text.shape[1])
        bert_output = self._dp(self._bert_model(text, attention_mask=attention_mask)[0])

        output_dict = {feat: self._linear_dict[feat](bert_output) for feat in self._linear_dict}
        return output_dict
