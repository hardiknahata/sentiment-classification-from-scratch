import numpy as np

class TextClassify:
  """
  The TextClassify class implements the Naive Bayes Classifier by representing
  a text document as a bag-of-words.

  Attributes
  ----------
  vocabulary : dict
      voacbulary of the corpus  

  vocabulary_size : int
      length of the vocabulary

  positive_words_count : int
      count of the positive class words

  positive_samples_count : int
      count of the positive class samples

  positive_samples_dict : dict
      dictionary of positive samples

  negative_words_count : int
      count of the negative class words

  negative_samples_count : int
      count of the negative class samples

  negative_samples_dict : dict
      dictionary of negative samples

  total_samples_count : int
      count of all samples

  positive_label : str
      label of the positive class

  negative_label : str
      label of the negative class
  """

  def __init__(self):
    '''
    Constructs all the necessary attributes for the TextClassify object.
    '''
    self.vocabulary = dict()
    self.vocabulary_size = 0
    self.positive_words_count = 0
    self.positive_samples_count = 0
    self.positive_samples_dict = dict()
    self.negative_words_count = 0
    self.negative_samples_count = 0
    self.negative_samples_dict = dict()
    self.total_samples_count = 0
    self.positive_label = '1'
    self.negative_label = '0'


  def add_to_dict(self, words, dictionary):
    '''
    Adds the given list of words to the given dictionary.
    Parameters:
        words (list): a list of words
        dictionary (dict): a dictionary 
    
    Returns: None
    '''
    for word in words:
      if word not in dictionary:
        dictionary[word] = 1
      else:
        dictionary[word] += 1  


  def get_score(self, words, label):
    '''
    Helper method for the score() method.
    Parameters:
        words (list): a list of words of the sample
        label (str): label of the sample
    
    Returns: 
        score (int): naive bayes probability for given label
    '''
    # case: positive label
    if label == self.positive_label:
      sentence_probability = 0
      class_prob = self.positive_samples_count / self.total_samples_count
      
      sentence_probability = np.log(class_prob) + sentence_probability
      
      denominator = self.positive_words_count + self.vocabulary_size
      
      for word in words:
        if word not in self.vocabulary:
          continue
        elif word in self.positive_samples_dict:
          word_count = self.positive_samples_dict[word]
        elif word in self.vocabulary and word not in self.positive_samples_dict:
          word_count = 0

        # laplace smoothing
        numerator = word_count + 1
        prob_word = numerator / denominator
        sentence_probability = np.log(prob_word) + sentence_probability

    # case: negative label    
    else:
      sentence_probability = 0
      class_prob = self.negative_samples_count / self.total_samples_count
      
      sentence_probability = np.log(class_prob) + sentence_probability
      
      denominator = self.negative_words_count + self.vocabulary_size

      for word in words:
        if word not in self.vocabulary:
          continue
        elif word in self.negative_samples_dict:
          word_count = self.negative_samples_dict[word]
        elif word in self.vocabulary and word not in self.negative_samples_dict:
          word_count = 0
          
        # laplace smoothing
        numerator = word_count + 1
        prob_word = numerator / denominator
        sentence_probability = np.log(prob_word) + sentence_probability
      
    return np.exp(sentence_probability)

      
  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    self.total_samples_count = len(examples)

    for sample in examples:
      idx, text, label = sample
      words = text.split()

      if label == self.positive_label:
        self.add_to_dict(words, self.positive_samples_dict)
        self.add_to_dict(words, self.vocabulary)
        self.positive_words_count += len(words)
        self.positive_samples_count += 1
      
      else:
        self.add_to_dict(words, self.negative_samples_dict)
        self.add_to_dict(words, self.vocabulary)
        self.negative_words_count += len(words)       
        self.negative_samples_count += 1
    
    self.vocabulary_size = len(self.vocabulary)


  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    likelihoods = dict()

    words = data.split()

    likelihoods[self.positive_label] = self.get_score(words, self.positive_label)
    likelihoods[self.negative_label] = self.get_score(words, self.negative_label)

    return likelihoods


  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    scores = self.score(data)
    if scores[self.positive_label] > scores[self.negative_label]:
      return self.positive_label
    else:
      return self.negative_label

  def featurize(self, data):
    """
    we use this format to make implementation of your TextClassifyImproved model more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    features = []
    words = data.split()

    for word in words:
      if word in self.vocabulary:
        features.append((word, True))
      else:
        features.append((word, False))
    
    return features


  def __str__(self):
    return "Naive Bayes Classifier (BOW baseline)"

