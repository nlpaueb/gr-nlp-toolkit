import string
from collections import Counter
from torch.utils.data import Dataset
import torch
import torch.nn as nn

class TextVectorizer:
    """
    Used to vectorize given text based on a learned vocabulary.
    After training the vocabulary on a corpus, the resulting encoding is:
    1                         : Padding
    [1 to max_vocab_size+1]   : tokens learnt in the vocab. (This could be smaller than the actual max number provided)
    len(vocab)-1              : [SOS] symbol
    len(vocab)                : OOV tokens
    """
    def __init__(self, mode):
        """
        vocab: dictionary (token --> index)
        idx2token: list (idx --> index)
        :param mode:
        """
        assert mode in {"word", "char"}
        self.vocab = dict()
        self.idx2token = []
        self.mode = mode

    def build_vocab(self, corpus, max_size=25000):
        """
        Builds the vocabulary from a corpus of sentences. The words get encoded by
        count of appearances in the data.
        :param corpus: A list of sentences as strings.
        :param max_size: The max size of words that can be encoded, excluding codes for <s> & OOV tokens.
        """
        counts = Counter()
        self.vocab["<pad>"] = 0
        self.idx2token.append("<pad>")
        idx = 1
        if self.mode == "word":
            # In the case of words, we remove punctuation and split on whitespaces
            for line in corpus:
                # Remove punctuation
                line = line.translate(str.maketrans("", "", string.punctuation))
                # Split the line in whitespaces to get the words
                tokens = line.split()
                # Update counts
                counts.update(tokens)
        # mode == "char"
        else:
            # Here we do not do any regularization, and split on every character.
            for line in corpus:
                tokens = [char for char in line]
                counts.update(tokens)
        # Add the most frequent tokens to the vocabulary.
        for (name, count) in counts.most_common(max_size):
            self.vocab[name] = idx
            self.idx2token.append(name)
            idx += 1
        # Add [SOS] token.
        self.vocab["<s>"] = idx
        self.idx2token.append("<s>")


    def encode_dataset(self, corpus):
        """
        Takes as input a corpus of sentences, generates source/target training pairs
        and encodes them based on the vocabulary. Then it returns the pairs as tuples of tensors.
        :param corpus: Array of sentences in the form of strings.
        :return: list of pairs of torch.LongTensor objects
        """
        # We start by tokenizing the corpus.
        tokenized_dataset = []
        for line in corpus:
            if self.mode == "word":
                # Strip punctuation and split on whitespaces.
                tokens = line.translate(str.maketrans("", "", string.punctuation)).split()
            else:
                # No regularization applied for characters.
                tokens = [char for char in line]
            # Also find the length of the longest sequence, taking into account the addition of a [SOS]/[EOS] symbol.
            tokenized_dataset.append(tokens)
        # Make source & target sentences and encode them based on the dictionary.
        source_vecs, target_vecs = [], []
        for sequence in tokenized_dataset:
            # Ignore strings that may be reduced to empty after stripping punctuation & whitespaces
            # (only happens if mode == "word")
            if not sequence:
                continue
            # Initialize source vectorized sentence with <s> token.
            source_vector = [self.vocab["<s>"]]
            target_vector = []
            for idx in range(len(sequence)-1):
                source_vector.append(self.vocab.get(sequence[idx], len(self.vocab)))
                target_vector.append(self.vocab.get(sequence[idx], len(self.vocab)))
            target_vector.append(self.vocab.get(sequence[-1], len(self.vocab)))
            # Add to sources/targets.
            source_vecs.append(source_vector)
            target_vecs.append(target_vector)

        """# Get the length for each sequence in the data
        source_lengths = torch.LongTensor(list(map(len, source_vecs)))
        target_lengths = torch.LongTensor(list(map(len, target_vecs)))"""
        # Convert data to LongTensors.
        for i in range(len(source_vecs)):
            source_vecs[i] = torch.LongTensor(source_vecs[i])
            target_vecs[i] = torch.LongTensor(target_vecs[i])
        # Pad & Sort sequences
        source_tensors = nn.utils.rnn.pad_sequence(source_vecs, batch_first=True)
        target_tensors = nn.utils.rnn.pad_sequence(target_vecs, batch_first=True)
        # Create Dataset object
        dataset = GreekDataset(source_tensors, target_tensors)
        # Return the Dataset & the sequence lengths (to be used for packing)
        return dataset

    def split_sequence(self, sequence):
        """
        Splits a sequence based on the tokenization mode configured, and returns it without indexing it.
        """
        if self.mode == "word":
            tokens = sequence.translate(str.maketrans("", "", string.punctuation)).split()
        else:
            tokens = [char for char in sequence]

        return sequence

    def input_tensor(self, sequence):
        """
        Takes a sentence and returns its encoding, based on the vocabulary, to be used for inference.
        :param sequence: (String) The sentence to be encoded.
        :return: Encoded sentence in form of a torch.Longtensor object.
        """
        if self.mode == "word":
            tokens = sequence.translate(str.maketrans("", "", string.punctuation)).split()
        else:
            tokens = [char for char in sequence]
        vectorized_input = []
        for token in tokens:
            vectorized_input.append(self.vocab.get(token, len(self.vocab)))

        # Convert to tensor
        vectorized_input = torch.LongTensor(vectorized_input)

        return vectorized_input


class GreekDataset(Dataset):

    def __init__(self, source_vecs, target_vecs):
        """
        Gets two arrays of source and target vectors and outputs a Dataset object of those arrays
        :param source_vecs: array of source vectors
        :param target_vecs: array of target vectors
        """
        self.n_samples = source_vecs.size(0)
        self.x_data = source_vecs
        self.y_data = target_vecs


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples