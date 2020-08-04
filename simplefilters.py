from basefilter import BaseFilter
from utils import read_classification_from_file, write_classification_to_file
from random import choice


class NaiveFilter(BaseFilter):
    """This filter classifies all the emails as OK"""
    def __init__(self):
        super(NaiveFilter, self).__init__()

    def train(self, training_corpus_path):
        self.dictionary = read_classification_from_file(training_corpus_path + '/!truth.txt')
        self.dictionary.fromkeys(self.dictionary, self.table[0])

    def test(self, prediction_corpus_path):
        write_classification_to_file(self.dictionary, prediction_corpus_path + '/!prediction.txt')


class ParanoidFilter(BaseFilter):
    """This filter classifies all the emails as SPAM"""
    def __init__(self):
        super(ParanoidFilter, self).__init__()

    def train(self, training_corpus_path):
        self.dictionary = read_classification_from_file(training_corpus_path + '/!truth.txt')
        self.dictionary.fromkeys(self.dictionary, self.table[1])

    def test(self, prediction_corpus_path):
        write_classification_to_file(self.dictionary, prediction_corpus_path + '/!prediction.txt')


class RandomFilter(BaseFilter):
    """This filter classifies all the emails by random"""
    def __init__(self):
        super(RandomFilter, self).__init__()

    def train(self, training_corpus_path):
        self.dictionary = read_classification_from_file(training_corpus_path + '/!truth.txt')
        self.dictionary = {x: choice(self.table) for x in self.dictionary}

    def test(self, prediction_corpus_path):
        write_classification_to_file(self.dictionary, prediction_corpus_path + '/!prediction.txt')
