from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch import nn
import torch



class ByT5Model(nn.Module):
    """
    A wrapper class for the T5 model for conditional generation

    Attributes:
        model (T5ForConditionalGeneration): The pre-trained ByT5 model
        tokenizer (AutoTokenizer): The tokenizer for the ByT5 model
    """

    def __init__(self, model_path = None, device = 'cpu'):
        """
        Initializes the ByT5Model with a pretrained T5 model and tokenizer.

        Args:
            model_path: The path to the pretrained model and tokenizer.
        """
        super(ByT5Model, self).__init__()
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model.to(self.device)

    def forward(self, text):
        """
        Performs inference to the ByT5 to generate the transliterated text

        Args:
            text: the input text in greeklish

        Returns:
            The output text in greek
        """
        self.model.eval()
        tokenized_text = self.tokenizer(text, return_tensors="pt").input_ids

        output = self.model.generate(tokenized_text.to(self.device), max_length=10000)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
