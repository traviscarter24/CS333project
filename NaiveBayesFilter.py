import numpy as np
import pandas as pd
import argparse
import re
from sklearn.feature_extraction.text import CountVectorizer #For data cleaning
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt') 
nltk.download('wordnet')


class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.vocabulary = None
        self.training_set= None
        self.test_set = None
        self.p_spam = None
        self.p_ham = None
        self.p_wi_spam = None
        self.p_wi_ham = None
        self.test_set_path = test_set_path
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['Label', 'SMS_Text'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['Label', 'SMS_Text'], encoding = 'utf-8')


    def data_cleaning(self):
        def parse(text):
            word = [word.lower() for word in text.split() if word.isalpha()]
            return word

        def clean_text(text):
            text = self.replace(text, self.replace_addresses)
            text = self.replace(text, self.replace_numbers)
            text = self.replace(text, self.replace_special_char)
            text = self.remove_duplicates(text)
            text = self.stem_and_lemmatize(text)
            return text

        self.training_set['SMS_Text'] = self.training_set['SMS_Text'].apply(lambda x: ' '.join(parse(clean_text(x))))
        self.test_set['SMS_Text'] = self.test_set['SMS_Text'].apply(lambda x: ' '.join(parse(clean_text(x))))
        self.vocabulary = set(word for sentence in self.training_set['SMS_Text'].apply(lambda x: x.split()) for word in sentence)

    def replace(self, text, replacement_function):
        return ' '.join(replacement_function(word) for word in text.split())

    def replace_addresses(self, word):
        if word.startswith('http') or '@' in word:
            return 'address'
        return word

    def replace_numbers(self, word):
        if word.isdigit() or (len(word) == 10 and word.replace('-', '').isdigit()):
            return 'number'
        return word

    def replace_special_char(self, word):
        special_characters = {'_', ',', '.', ';', ':', '?', '!', '"','$', '#', '%', '&', '*', '(', ')', '[', ']', '{', '}', '/', '+', '-', '='}
        if any(char in word for char in special_characters):
            return 'special_char'
        return word

    def remove_duplicates(self, text):
        word = text.split()
        unique_word = set(word)
        return ' '.join(unique_word)

    def stem_and_lemmatize(self, text):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        word = word_tokenize(text)
        stemmed_word = [stemmer.stem(lemmatizer.lemmatize(word)) for word in word]
        return ' '.join(stemmed_word)

    def fit_bayes(self):
        self.p_spam = np.sum(self.training_set['Label'] == 'spam') / len(self.training_set)
        self.p_ham = 1 - self.p_spam

        n_spam = self.training_set[self.training_set['Label'] == 'spam']['SMS_Text'].apply(len).sum()
        n_ham = self.training_set[self.training_set['Label'] == 'ham']['SMS_Text'].apply(len).sum()
        n_vocabulary = len(self.vocabulary)

        alpha = 0.25

        self.p_wi_spam = {}
        self.p_wi_ham = {}

        for word in self.vocabulary:
            n_wi_spam = self.training_set[self.training_set['Label'] == 'spam']['SMS_Text'].apply(lambda x: x.split().count(word)).sum()
            n_wi_ham = self.training_set[self.training_set['Label'] == 'ham']['SMS_Text'].apply(lambda x: x.split().count(word)).sum()
            self.p_wi_spam[word] = (n_wi_spam + alpha) / (n_spam + (alpha * n_vocabulary))
            self.p_wi_ham[word] = (n_wi_ham + alpha) / (n_ham + (alpha * n_vocabulary))

    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()

    def sms_classify(self, message):
        word = [word.lower() for word in message.split() if word.isalpha()]
        p_spam_given_message = np.log(self.p_spam) + np.sum([np.log(self.p_wi_spam.get(word, 1e-10)) for word in word])
        p_ham_given_message = np.log(self.p_ham) + np.sum([np.log(self.p_wi_ham.get(word, 1e-10)) for word in word])

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'

    def classify_test(self):
        correct = 0
        for i in range(len(self.test_set)):
            predict = self.sms_classify(self.test_set["SMS_Text"][i])
            if predict == self.test_set['Label'][i]:
                correct += 1
        accuracy = (correct/len(self.test_set))
        return accuracy*100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
