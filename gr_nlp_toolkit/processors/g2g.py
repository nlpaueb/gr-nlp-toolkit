from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.models import g2g_RBNLM_model
from gr_nlp_toolkit.domain.textVectorizer import TextVectorizer
from gr_nlp_toolkit.models.g2g_RBNLM_model import LanguageModel
from gr_nlp_toolkit.models.g2g_transformer_model import ByT5Model
import torch
import pickle


class G2G(AbstractProcessor):

    def __init__(self, mode = 'LSTM', model_path = None, tokenizer_path = None, device = 'cpu'):
        
        self.mode = mode

        if self.mode == 'LSTM':
            input_size = 120
            embed_size = 32
            hidden_size = 512
            output_size = 120
            
            # Load and initialize the LSTM model
            self.model = g2g_RBNLM_model.LSTM_LangModel(input_size, embed_size, hidden_size, output_size)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

            # Load and initialize the tokenizer
            self.text_vectorizer = TextVectorizer("char")
            with open(tokenizer_path, "rb") as file:
                self.text_vectorizer = pickle.load(file)

            # Initialize the LanguageModel
            self.LM = LanguageModel(self.text_vectorizer, self.model)

        elif self.mode == 'transformer':
            self.model = ByT5Model(model_path)

    def __call__(self, doc: Document) -> Document:
        # predict
        if(self.mode == 'LSTM'):
            doc.text = self.LM.translate([doc.text], 5)[0]
        elif(self.mode == 'transformer'):
            doc.text = self.model(doc.text)
        return doc
    




