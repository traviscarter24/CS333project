import unittest
from NaiveBayesFilter import NaiveBayesFilter
import numpy as np

class TestNaiveBayesFilterIntegration(unittest.TestCase):
    def setUp(self):
        self.classifier = NaiveBayesFilter('test.csv')

    def test_integration_1(self):
        self.classifier.read_csv()
        self.classifier.data_cleaning()
        self.classifier.fit_bayes()
        self.assertEqual(self.classifier.sms_classify('WINNER!! You have been selected for a cash prize!'), 'spam')

    def test_integration_2(self):
        self.classifier.read_csv()
        self.classifier.data_cleaning()
        self.classifier.fit_bayes()
        self.assertEqual(self.classifier.sms_classify('Hey, how are you?'), 'ham')

    def test_integration_3(self):
        self.classifier.read_csv()
        self.classifier.data_cleaning()
        self.classifier.fit_bayes()
        self.assertEqual(self.classifier.sms_classify('Call now for a free quote'), 'spam')

    def test_integration_4(self):
        self.classifier.read_csv()
        self.classifier.data_cleaning()
        self.classifier.fit_bayes()
        self.assertEqual(self.classifier.sms_classify('Can we meet tomorrow?'), 'ham')

    def test_integration_5(self):
        self.classifier.read_csv()
        self.classifier.data_cleaning()
        self.classifier.fit_bayes()
        self.assertEqual(self.classifier.sms_classify('Congratulations! You won a free trip to Hawaii!'), 'spam')

if __name__ == '__main__':
    unittest.main()
