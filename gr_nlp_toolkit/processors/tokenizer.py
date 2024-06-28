from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Dataset

from gr_nlp_toolkit.domain.dataset import DatasetImpl
from gr_nlp_toolkit.domain.document import Document
from gr_nlp_toolkit.processors.abstract_processor import AbstractProcessor

import unicodedata

from gr_nlp_toolkit.domain.token import Token

from transformers import AutoTokenizer

tokenizer_greek = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')


def strip_accents_and_lowercase(s: str) -> str:
    """
    Strips accents from a string and converts it to lowercase.

    Args:
        s: A string from which to strip accents.

    Returns:
        A new string with accents removed and converted to lowercase.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()


def create_ids(text: str) -> List[int]:
    """
    Encodes a given text into a list of token IDs using a tokenizer.

    Args:
        text: A string to encode.

    Returns:
        A list of token IDs.
    """
    return tokenizer_greek.encode(text)


def create_text(ids: List[int]) -> List[int]:
    """
    Decodes a list of token IDs back into a text string.

    Args:
        ids: A list of token IDs to decode.

    Returns:
        A decoded string with special tokens skipped.
    """
    return tokenizer_greek.decode(ids, skip_special_tokens=True)


def convert_to_tokens(input_ids: List[int]) -> List[str]:
    """
    Converts a list of token IDs into their corresponding token strings.

    Args:
        input_ids: A list of token IDs.

    Returns:
        A list of token strings with special tokens skipped.
    """
    return tokenizer_greek.convert_ids_to_tokens(input_ids, skip_special_tokens=True)


def remove_special_tokens(input_ids: List[int]) -> List[int]:
    """
    Removes special tokens from a list of token IDs.

    Args:
        input_ids: A list of token IDs.

    Returns:
        A new list of token IDs with special tokens removed.
    """
    input_ids_without_special_tokens = []
    for input_id in input_ids:
        if input_id not in tokenizer_greek.all_special_ids:
            input_ids_without_special_tokens.append(input_id)
    return input_ids_without_special_tokens



def create_mask_and_tokens(input_tokens: List[str], input_ids: List[int]) -> Tuple[List[str], List[Token], Dict]:
    """
    Creates a mask, tokens, and subword-to-word mapping from input tokens and IDs.

    Args:
        input_tokens: A list of input token strings.
        input_ids: A list of input token IDs.

    Returns:
        A tuple containing:
        - A list of booleans indicating whether each token is a subword.
        - A list of Token objects.
        - A dictionary mapping subword indices to word indices.
    """
    mask = []
    tokens = []
    subword2word = {}

    word = 0
    # for each token
    for j, input in enumerate(zip(input_tokens, input_ids), 1):
        t = input[0]
        i = input[1]
        # it isn't a sub-word
        if not t.startswith("##"):
            # create a token object
            tokenObj = Token([t])
            tokenObj.ids.append(i)
            tokens.append(tokenObj)
            mask.append(True)
            word = word + 1
        else:

            # add sub-words to token
            tokenObj.subwords.append(t)
            tokenObj.ids.append(i)
            mask.append(False)
        subword2word[j] = word

    # create text
    for token in tokens:
        token.text = create_text(token.ids)

    # Adding a 0-0 mapping to subword2word
    subword2word[0] = 0

    return mask, tokens, subword2word


def create_dataset_and_dataloader(input_ids) -> Tuple[Dataset, DataLoader]:
    """
    Creates a dataset and dataloader from input IDs.

    Args:
        input_ids: A list of input token IDs.

    Returns:
        A tuple containing:
        - A Dataset object.
        - A DataLoader object.
    """
    dataset = DatasetImpl([input_ids])
    dataloader = DataLoader(dataset)
    return dataset, dataloader



class Tokenizer(AbstractProcessor):
    """
    Tokenizer class that takes a document as an input with the text field set, tokenizes and returns a document with
    all fields set
    """

    def __call__(self, doc: Document) -> Document:
        """
        Processes a document by tokenizing its text and setting relevant fields.

        Args:
            doc: A Document object with the text field set.

        Returns:
            A Document object with the following fields set:
            - text: The original text stripped of accents and converted to lowercase.
            - input_ids: List of token IDs created from the text.
            - token_mask: List of booleans indicating whether each token is a subword.
            - tokens: List of Token objects.
            - subword2word: Dictionary mapping subword indices to word indices.
            - dataset: A Dataset object created from the input IDs.
            - dataloader: A DataLoader object created from the dataset.
        """
        # get document's text and strip accent and lowercase
        doc.text = strip_accents_and_lowercase(doc.text)
        # create ids
        doc.input_ids = create_ids(doc.text)
        # create mask and tokens
        doc.token_mask, doc.tokens, doc.subword2word = create_mask_and_tokens(convert_to_tokens(doc.input_ids),
                                                                        remove_special_tokens(doc.input_ids))

        # create dataloader
        doc.dataset, doc.dataloader = create_dataset_and_dataloader(doc.input_ids)
        return doc
