import pytorch_wrapper.functional as pwF

from torch import nn


class NERBERTModel(nn.Module):

    def __init__(self, bert_model, model_output_size, dp):
        super(NERBERTModel, self).__init__()
        self._bert_model = bert_model
        self._dp = nn.Dropout(dp)
        self._output_linear = nn.Linear(768, model_output_size)

    def forward(self, text, text_len):
        attention_mask = pwF.create_mask_from_length(text_len, text.shape[1])
        return self._output_linear(self._dp(self._bert_model(text, attention_mask=attention_mask)[0]))