"""
Author: Hardik Nahata
Email: nahata.h@northeastern.edu
"""

"""
Citations:
Bing Liu's sentiment lexicon of positive and negative words
NLTK word tokenizer
NLTK stopwords list
"""

import sys
from src.utils import Metrics as metrics
from src.utils import EventsIO as io
from src.models import naive_bayes
from src.models import logistic_regression

def main():
  training = sys.argv[1]
  testing = sys.argv[2]

  # train samples
  samples = io.generate_tuples_from_file(training_file_path = training)
  # test samples
  test_samples = io.generate_tuples_from_file(training_file_path = testing)

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
  print(f"Precision: {metrics.precision(gold_labels, pred_labels)}\n")
  print(f"Recall: {metrics.recall(gold_labels, pred_labels)}\n")
  print(f"F1 Score: {metrics.f1(gold_labels, pred_labels)}\n")

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
  print(f"Precision: {metrics.precision(gold_labels, pred_labels)}\n")
  print(f"Recall: {metrics.recall(gold_labels, pred_labels)}\n")
  print(f"F1 Score: {metrics.f1(gold_labels, pred_labels)}\n")  

  
if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
