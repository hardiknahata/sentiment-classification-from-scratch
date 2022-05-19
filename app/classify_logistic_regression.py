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
from src.utils import EventsIO
from src.models import logistic_regression


class SentimentClassificationLR:

  def __init__(self) -> None:
    self.io = EventsIO()
    self.train_file_path = "src/training_files/train_file.txt"
    self.positive_label = '1'
    self.negative_label = '0'

    
  def predict(self, sample):
    training = "src/training_files/train_file.txt"

    # train samples
    samples = self.io.generate_tuples_from_file(training_file_path = training)

    # Naive Bayes Classifier
    classifier = logistic_regression.TextClassify()
    classifier.train(samples)

    prediction = classifier.classify(sample)
    
    if prediction == self.positive_label:
      return "POSITIVE"
    else:
      return "NEGATIVE"

  
if __name__ == "__main__":
  text = "i am not happy with the hotel service"
  prediction = SentimentClassificationLR().predict(text)
  print(prediction)

