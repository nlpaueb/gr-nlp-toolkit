import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gr_nlp_toolkit.configs.dictionary_tables import greeklish_to_greek_intonated

class LSTM_LangModel(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super(LSTM_LangModel, self).__init__()
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, h0=None, c0=None):
        input_embedded = self.embed(x)
        if h0 is None and c0 is None:
            output_lstm, (h, c) = self.lstm(input_embedded)
        else:
            output_lstm, (h, c) = self.lstm(input_embedded, (h0, c0))
        output = self.dropout(output_lstm)
        output = self.dense(output)
        return output, h, c
    
class State():

    def __init__(self, translated, remaining, out, hidden, score):
        """
        Container class for the attributes of each candidate replacement.
        :param translated: (list) List of tokens already translated to the desired language.
        :param remaining: (str) The remaining sentence to be translated.
        :param out: (tensor) The last output of the LSTM for that particular state. Contains
                    logits that will become probabilities with the application of a softmax.
        :param hidden: (tuple) Contains the hidden states of the candidate.
        :param score: (float) the score given to the translation based on the language model.
        """
        self.translated = translated
        self.remaining = remaining
        self.hidden = hidden
        self.output = out
        self.score = score

    def __eq__(self, other):
        """
        Equality operator, needed for eliminating duplicate states.
        """
        if isinstance(other, State):
            if (self.translated == other.translated and
                    self.remaining == other.remaining and
                    self.score == other.score):
                return True

        return False

class LanguageModel:

    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        self.mode = vectorizer.mode
        # Use the log version of Softmax to sum scores instead of multiplying them and avoid decay.
        self.softmax = nn.LogSoftmax(dim=1)

    def load_model(self, path):
        """
        Load a pre-trained model as a state dictionary.
        """
        self.model.load_state_dict(torch.load(path))

    def translate(self, sentences, beams):
        """
        Takes a list of sentences and translates them.
        :param sentences: (list) Sentences you want to translate
        :param beams: (int) The number of parameters
        :return: (list) Translated sentences
        """
        # Don't forget to put the model in eval mode.
        self.model.eval()
        with torch.no_grad():
            translated_sentences = []
            for sentence in sentences:
                translated = []
                remaining = sentence
                # We start with the first state, with a score of 1
                # The format of a state is: (translated_sent, remaining_sent, (h0, c0), score)
                # --------------------------------------------------------------------------------
                # First, we need to "prep" the char model. This is done by feeding the network with the
                # <s> token and saving the output hidden states for the initial State() object.
                start_input = self.vectorizer.input_tensor("<s>")
                out, h_n, c_n = self.model(start_input, None, None)
                # The score of the initial state in 0, because we use LogSoftmax instead of regular Softmax.
                initial_state = State(translated, remaining, out, (h_n, c_n), 0)
                states = [initial_state]
                for i in range(len(sentence)):
                    candidates = []
                    # Look through the current states.
                    for state in states:
                        # Produce the next-char-candidates from each state, along with their probabilities.
                        new_states = self.get_candidates(state)
                        candidates += new_states

                    # Remove any duplicate candidates
                    #print([candidate.translated for candidate in candidates])
                    candidates_set = []
                    [candidates_set.append(cand) for cand in candidates if cand not in candidates_set]
                    candidates = candidates_set
                    # Get the best states, according to the number of beams in the search.
                    best_candidates = []
                    for j in range(beams):
                        if candidates:
                            # Probabilities of each candidate new state.
                            probabilities = [cand.score for cand in candidates]
                            # Get the best candidate and remove from the list
                            best_cand = candidates.pop(probabilities.index(max(probabilities)))
                            # Add that candidate to the list of best candidates
                            best_candidates.append(best_cand)
                    # Make the list of the best candidates the new states.
                    states = best_candidates

                # Once the sentence is over, the state with the highest probability is the best translation.
                probs = [state.score for state in states]
                # Extract the sentence from the state
                sent = states[probs.index(max(probs))]
                # Convert the list of translated tokens to a sentence.
                translation = ""
                for i in sent.translated:
                    translation += i

                translated_sentences.append(translation)
            return translated_sentences

    def get_candidates(self, state):
        # If the state is already a final state (no remaining text to translate),
        # it returns only itself as a candidate.
        if not state.remaining:
            return [state]

        # If it is not a final state, generate the next candidate states
        candidates = []

        # Look at both of the first two characters in the translated sentence, as some greek
        # characters may be represented with 2 characters in Greeklish.
        for length in [1, 2]:
            if len(state.remaining) >= length:
                # Fetch the valid replacements from the dictionary.
                if length == 2:
                    token = state.remaining[0] + state.remaining[1]
                    replacements = greeklish_to_greek_intonated.get(token, [])
                else:
                    # If the look-up is a miss (e.g. the token is a space, a number or punctuation),
                    # return the token itself.
                    token = state.remaining[0]
                    replacements = greeklish_to_greek_intonated.get(token, [token])

                # For each candidate replacement, get the probability from the LM
                for item in replacements:
                    h_n, c_n = state.hidden[0], state.hidden[1]
                    out = state.output
                    score = state.score
                    for token in item:
                        # Apply softmax to the model's output and get the prob based on the index from vocab
                        probs = self.softmax(out)
                        idx = self.vectorizer.vocab.get(token, len(self.vectorizer.vocab))
                        # Update score.
                        score = score + probs[0][idx].item()
                        # Feed the token to the model to get the next output and hidden states
                        input = self.vectorizer.input_tensor(token)
                        out, h_n, c_n = self.model(input, h_n, c_n)

                    translated_tokens = [token for token in item]

                    new_candidate = State(state.translated+translated_tokens,
                                          state.remaining[length:],
                                          out, (h_n, c_n), score)
                    candidates.append(new_candidate)

        return candidates