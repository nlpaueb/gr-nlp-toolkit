class SentenceProcess:

    def __init__(self, sentence: str):
        self.sentence = sentence

    def split_sentence(self) -> list:
        """Splits the sentence into tokens based on space.
        :return: list of strings
        """
        return self.sentence.split(" ")
