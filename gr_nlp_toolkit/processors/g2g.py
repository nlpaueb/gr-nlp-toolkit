from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.models import g2g_RBNLM_model
from gr_nlp_toolkit.domain.textVectorizer import TextVectorizer
from gr_nlp_toolkit.models.g2g_RBNLM_model import LanguageModel
from gr_nlp_toolkit.models.g2g_transformer_model import ByT5Model
import torch
import pickle

def detect_language(text):
    """
    Checks whether the majority of the letters in the input text are in the greek or the latin script
    It is used to identify whether the text is in greek or greeklish (latin script), in order to skip unnecessary conversions.

    Args:
        text (str): The input text

    Returns:
        script (str): The dominant script
    """
    # Filter out non-letter characters
    valid_characters = [char for char in text if char.isalpha()]
    
    # Count Greek and English letters
    greek_count = sum(1 for char in valid_characters if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF')
    english_count = sum(1 for char in valid_characters if '\u0041' <= char <= '\u005A' or '\u0061' <= char <= '\u007A')
    
    script = "greek" if greek_count >= english_count else "latin"
    return script


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
        self.device = torch.device(device)
        
        if self.mode == 'LSTM':
            # Define the model parameters (more info: https://aclanthology.org/2024.lrec-main.1330/)
            input_size = 120
            embed_size = 32
            hidden_size = 512
            output_size = 120
            
            # Load and initialize the LSTM model
            self.beam_size = 5
            self.model = g2g_RBNLM_model.LSTM_LangModel(input_size, embed_size, hidden_size, output_size)
            

            # Load and initialize the tokenizer
            self.text_vectorizer = TextVectorizer("char")
            
            if(model_path is not None):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

            
            if(tokenizer_path is not None):
                with open(tokenizer_path, "rb") as file:
                    self.text_vectorizer = pickle.load(file)

            # Initialize the LanguageModel
            self.LM = LanguageModel(self.text_vectorizer, self.model, device=self.device)
                    

        elif self.mode == 'transformer':
            self.model = ByT5Model(model_path, device=self.device)
            self.model.eval()


    def __call__(self, doc: Document) -> Document:
        """
        Processes a document to perform Greeklish to Greek conversion.

        Args:
            doc (Document): The document to process.

        Returns:
            Document: The document with text converted using the specified model.
        """

        # If the text is in already in greek, skip the g2g conversion
        if(detect_language(doc.text) == 'greek'):
            return doc
        
        

        # Perform G2G conversion based on the mode
        if(self.mode == 'LSTM'):
            doc.text = self.LM.translate([doc.text], self.beam_size)[0]
        elif(self.mode == 'transformer'):
            doc.text = self.model(doc.text)

        return doc