from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch import nn

class ByT5Model(nn.Module):
    def __init__(self, model_path = None):
        super(ByT5Model, self).__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, text):
        self.model.eval()
        tokenized_text = self.tokenizer(text, return_tensors="pt").input_ids
        output = self.model.generate(tokenized_text, max_length=200)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)