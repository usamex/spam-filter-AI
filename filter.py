from collections import Counter
from functools import reduce
from utils import read_classification_from_file, write_classification_to_file
import email
import re
import os


class MyFilter:
    '''
        This filter uses Naive Bayes method to classify ham and spam emails.
        Studied from Wikipedia/Naive Bayes spam filtering technique created by Paul Graham.
        http://www.paulgraham.com/spam.html
    '''

    def __init__(self):
        self.decision_table = ['OK', 'SPAM']
        self.train_files_dict = dict()
        self.test_files_result_dict = dict()
        self.init_spam_likelihood = 0.4  # This has been selected by empirical methods
        self.words_counter = Counter()  # It holds the words' counter
        self.spam_words_counter = Counter()  # It holds the spam words' counter
        self.ham_words_counter = Counter()  # It holds the ham words' counter
        self.vocabulary = set()  # It holds all the vocabulary
        self.total_ham_emails = 0  # It holds total ham emails in the training set
        self.total_spam_emails = 0  # It holds total ham emails in the training set
        self.word_spaminess = dict()  # It holds the spaminess of the words

    def train(self, train_dir):
        self.train_files_dict = read_classification_from_file(train_dir + '/!truth.txt')
        total_emails = len(self.train_files_dict)

        for file in self.train_files_dict:
            train_file_path = train_dir
            train_file_path += '/' + file
            mail = self.get_email(train_file_path)
            mail_words = self.get_email_message(mail)
            mail_unique_words = set(mail_words)
            """Counting spam and ham word appearances"""
            if self.train_files_dict[file] == self.decision_table[1]:
                self.spam_words_counter.update(mail_words)
                self.total_spam_emails += 1
            else:
                self.ham_words_counter.update(mail_words)

            self.words_counter.update(mail_words)
            self.vocabulary.update(mail_unique_words)

        self.total_ham_emails = total_emails - self.total_spam_emails

        """Computing the probability that a message containing a given word is spam."""
        for word in self.vocabulary:
            if self.ham_words_counter.get(word, 0) == 0 and self.spam_words_counter.get(word, 0) > 0:
                self.word_spaminess[word] = 0.99
            elif self.ham_words_counter.get(word, 0) > 0 and self.spam_words_counter.get(word, 0) == 0:
                self.word_spaminess[word] = 0.01
            else:
                spam_likelihood = self.spam_words_counter.get(word, 0) / self.total_spam_emails
                ham_likelihood = self.ham_words_counter.get(word, 0) / self.total_ham_emails
                self.word_spaminess[word] = max(spam_likelihood / (spam_likelihood + ham_likelihood), 0.01)

    def test(self, test_dir):
        test_files = os.listdir(test_dir)
        for file in test_files:
            test_file_path = test_dir
            test_file_path += '/' + file
            mail = self.get_email(test_file_path)
            mail_words = self.get_email_message(mail)
            word_ratings = []
            """Setting word spam ratings"""
            for word in mail_words:
                if word in self.vocabulary:
                    word_ratings.append(self.word_spaminess.get(word, self.init_spam_likelihood))
                else:
                    word_ratings.append(self.init_spam_likelihood)

            """Paul Graham - A Plan for Spam method."""
            if len(word_ratings) == 0:
                self.test_files_result_dict[file] = self.decision_table[1]
                continue
            elif len(word_ratings) >= 20:
                """To avoid rounding to zero"""
                word_ratings.sort()
                word_ratings = word_ratings[:10] + word_ratings[-10:]

            """Combining individual probabilities of that the message containing a spam word"""
            """I'm assuming that the words present in the message are independent events. 
            So that's why I'm multiplying all the word ratings."""

            """Product of all word_spaminess in the message."""
            spam_rating_product = reduce(lambda x, y: x * y, word_ratings)
            """Product of all word_haminess in the message."""
            ham_rating_product = reduce(lambda x, y: x * y, map(lambda x: 1.0 - x, word_ratings))
            result = spam_rating_product / (spam_rating_product + ham_rating_product)

            """After the email's spam probability is computed over all words in the email, 
            and if the total exceeds a certain threshold, the filter will mark the email as a spam."""
            if result >= 0.95:
                self.test_files_result_dict[file] = self.decision_table[1]
            else:
                self.test_files_result_dict[file] = self.decision_table[0]

        write_classification_to_file(self.test_files_result_dict, test_dir + '/!prediction.txt')

    @staticmethod
    def get_email(file_path):
        """Opens mail."""
        fp = open(file_path, 'r', encoding='ISO-8859-1')
        mail = email.message_from_file(fp)
        fp.close()
        return mail

    @staticmethod
    def strip_html(html_string):
        """ `Regular expressions are a tool that is insufficiently sophisticated to understand the constructs employed
        by HTML. HTML is not a regular language and hence cannot be parsed by regular expressions.` via Stack Overflow
        """
        stripped_string = str()
        delete = False
        for letter in html_string:
            if letter == '<':
                delete = True
                continue
            if letter == '>':
                delete = False
                continue
            if delete:
                continue
            stripped_string += letter
        return stripped_string

    def trim_email_payload(self, string):
        """The regex can find $100,000, zzzz-ssss and xyz type of strings."""
        string = string.lower()
        trimmed_string_pattern = re.compile(r'\$?\d*(?:[.,]\d+)+|\w+-\w+|\w+', re.U)
        stripped_string = self.strip_html(string)
        word_list = list(filter(lambda s: len(s) > 2, re.findall(trimmed_string_pattern, stripped_string)))
        return word_list

    def get_email_message(self, mail_object):
        """
        :param mail_object is an email.Message object. Its message wanted to be read
        :return returns the list of email words.

        ISO-8859-1 was (according to the standards at least) the default encoding of documents delivered via HTTP
        with a MIME type beginning with "text/" (HTML5 changed this to Windows-1252). ~Wikipedia
        """

        if not mail_object.is_multipart():
            string = str(mail_object.get_payload(decode=True), 'ISO-8859-1')
            payload = str()
            if len(string) == 0:
                payload += mail_object.get_payload()
            else:
                payload += string
        else:
            payload = self.get_payload_string(mail_object)

        clean_payload = self.trim_email_payload(payload)
        return clean_payload

    def get_payload_string(self, mail_object):
        """In case the email is a MIME type of email, the get_payload function will return a list.
        I created this function to get the email message as a string."""

        mail_string = str()
        payload_list = mail_object.get_payload()
        for payload in payload_list:
            if payload.is_multipart():
                mail_string += self.get_payload_string(payload)
            else:
                """Decoding sometimes returns bytes-type of object. That's why I decode it as ISO-8859-1."""
                string = str(payload.get_payload(decode=True), 'ISO-8859-1')
                if len(string) == 0:
                    mail_string += payload.get_payload()
                else:
                    mail_string += string
        return mail_string
