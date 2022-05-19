"""
Author: Hardik Nahata
Email: nahata.h@northeastern.edu
"""

"""
CITATION:
Bing Liu's sentiment lexicon of positive and negative words
NLTK word tokenizer
NLTK stopwords list
"""

import sys
from src.utils import Metrics
from src.utils import EventsIO
from src.models import naive_bayes
from src.models import logistic_regression

class SentimentClassification:

  def __init__(self) -> None:
    self.io = EventsIO()
    self.metrics = Metrics()
    self.training_path = "src/training_files/train_file.txt"
    self.testing_path = "src/training_files/dev_file.txt"
    
  def run(self):
    # train samples
    samples = self.io.generate_tuples_from_file(training_file_path = self.training_path)
    # test samples
    test_samples = self.io.generate_tuples_from_file(training_file_path = self.testing_path)

    # Naive Bayes Classifier - Baseline
    classifier = naive_bayes.TextClassify()
    print('\n')
    print(classifier)
    classifier.train(samples)

    # Performance metrics
    gold_labels = []
    pred_labels = []
    for sample in test_samples:
        idx, text, label = sample
        gold_labels.append(label)
        pred_labels.append(classifier.classify(text))

    # report precision, recall, f1
    print(f"Precision: {self.metrics.precision(gold_labels, pred_labels)}\n")
    print(f"Recall: {self.metrics.recall(gold_labels, pred_labels)}\n")
    print(f"F1 Score: {self.metrics.f1(gold_labels, pred_labels)}\n")

    # Logistic Regression Classifier with 4 features and preprocessing
    improved = logistic_regression.TextClassify()
    print(improved)
    improved.train(samples)

    # Performance metrics
    gold_labels = []
    pred_labels = []
    for sample in test_samples:
        idx, text, label = sample
        gold_labels.append(label)
        pred_labels.append(improved.classify(text))

    ## report final precision, recall, f1 (for your best model)
    print(f"Precision: {self.metrics.precision(gold_labels, pred_labels)}\n")
    print(f"Recall: {self.metrics.recall(gold_labels, pred_labels)}\n")
    print(f"F1 Score: {self.metrics.f1(gold_labels, pred_labels)}\n")  

  
if __name__ == "__main__":

  SentimentClassification().run()
