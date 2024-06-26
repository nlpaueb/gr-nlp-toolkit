from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.models import g2g_RBNLM_model
from gr_nlp_toolkit.domain.textVectorizer import TextVectorizer
from gr_nlp_toolkit.models.g2g_RBNLM_model import LanguageModel
from gr_nlp_toolkit.models.g2g_transformer_model import ByT5Model
import torch
import pickle


class G2G(AbstractProcessor):
    """
        Greeklsih to Greek (G2G) processor class.

        This class performs G2G conversion using either an LSTM-based model or a transformer-based model.
        It initializes the model, loads the necessary components, and provides functionality to process documents
        and convert text using the specified mode.
    """

    def __init__(self, mode = 'LSTM', model_path = None, tokenizer_path = None, device = 'cpu'):

        """
        Initializes the G2G class with the specified parameters.

        Args:
            mode (str, optional): The mode of the model, either 'LSTM' or 'transformer'. Defaults to 'LSTM'.
            model_path (str, optional): Path to the pre-trained model. Defaults to None.
            tokenizer_path (str, optional): Path to the tokenizer for LSTM mode. Defaults to None.
            device (str, optional): Device to load the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """

        self.mode = mode
        
        if self.mode == 'LSTM':
            input_size = 120
            embed_size = 32
            hidden_size = 512
            output_size = 120
            
            # Load and initialize the LSTM model
            self.beam_size = 5
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
        """
        Processes a document to perform Greeklish to Greek conversion.

        Args:
            doc (Document): The document to process.

        Returns:
            Document: The document with text converted using the specified model.
        """

        # Perform G2G conversion based on the mode
        if(self.mode == 'LSTM'):
            doc.text = self.LM.translate([doc.text], self.beam_size)[0]
        elif(self.mode == 'transformer'):
            doc.text = self.model(doc.text)
        return doc
    




