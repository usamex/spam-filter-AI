class BaseFilter:
    def __init__(self):
        self.table = ['OK', 'SPAM']
        self.dictionary = dict()

    def test(self, prediction_corpus_path):
        pass

    def train(self, training_corpus_path):
        pass
