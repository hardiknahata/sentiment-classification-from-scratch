import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from ..utils import EventsIO

class TextClassify:
  """
  The class TextClassify class implements the Logistic Regression Classifier by representing
  a text document as a bag-of-words.

  Features (CS6120 requirements satisfied)
  --------
  - Bag of Wrords
  - Count of Positive and Negative words
  - Logarithm of review length
  - Count of first and second person pronouns

  Preprocesssing
  ----------
  - lowecase text
  - stopwords removed
  - nltk word tokenizer

  Attributes
  ----------
  reviews : list
      list of all reviews the model is trained on

  labels : list
      list of all corresponding reviews labels

  word_freq : dict
      word frequency of the complete trained corpus

  vocab : list
      vocabulary of the corpus

  bow : list[list]
      a list of list containing the vectors for each review

  bias : int
      bias is an adjustable, numerical term added to weighted sum of inputs and weights that can 
      increase classification model accuracy

  epochs : int
      number of epochs is a hyperparameter that defines the number times that the learning algorithm 
      will work through the entire corupus

  weights : dict
      the learned model weights which will help classify new samples

  lr : int
      the learning rate of the gradient descent algorithm

  positive_label : str
      label of the positive class

  negative_label : str
      label of the negative class

  pos_words : list
      list of positive words taken from Bing Liu's sentiment lexicon

  neg_words : list
      list of negative words taken from Bing Liu's sentiment lexicon

  pronouns : list
      list of common first and second person pronouns

  stopwords : list
      list of stopwords taken from NLTK libary
  """

  def __init__(self):
    '''
    Constructs all the necessary attributes for the TextClassify object.
    '''    
    self.io = EventsIO()
    self.reviews = []
    self.labels = []
    self.word_freq = {}
    self.vocab = []
    self.bow = []
    self.bias = 0
    self.epochs = 100
    self.weights = []
    self.lr = 0.0001
    self.positive_label = '1'
    self.negative_label = '0'
    self.pos_words = self.io.load_lexicon('src/training_files/positive-words.txt')
    self.neg_words = self.io.load_lexicon('src/training_files/negative-words.txt')    
    self.pronouns = ['I', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves']
    self.stopwords = ['doesn', "hasn't", 'only', 'his', 'am', 'which', 'theirs', 'them', "she's", 'at', 'up', 'own', 'just', 'you', 'what', 'wouldn', 'once', 'any', 's', 'below', 'yourself', 'or', 'is', 'further', 'here', 'those', 'an', 'of', 'all', 'nor', 'won', "should've", "isn't", 'ain', 'are', 'yours', "that'll", 'both', 'and', 'needn', 'some', "mightn't", "wouldn't", 'when', 'than', 'couldn', "didn't", 'the', 'their', 'under', 'very', 'hasn', 'haven', 'aren', "it's", 'over', 'were', "don't", "shouldn't", 'does', 'it', 'into', 'd', 'ourselves', 'more', "haven't", 'too', "doesn't", 'who', 'did', 'itself', 'off', 'as', 'with', 'had', 'in', 'most', 't', 'do', 'now', 'until', 're', "you'll", 'has', 'during', 'on', 'where', 'y', 'didn', 'few', "you're", 'ours', "mustn't", "wasn't", 'o', 'while', 'whom', 'to', 'between', 'ma', 'himself', 'then', 'no', 'myself', "shan't", 'mightn', 'being', 'having', 'wasn', 'again', 'but', 'against', 'each', 'these', 'before', 'her', 'such', "couldn't", "hadn't", 'we', 'out', 'hers', 'don', 'there', 'm', 'mustn', 'my', 'through', "won't", 'i', 'can', 'other', 'this', "you've", 'for', 'our', 'your', 'doing', 'how', 'was', "aren't", 'shouldn', 'from', 'that', 'been', 'about', 'hadn', 'shan', 'if', 'a', 'after', 'he', 'themselves', 'have', 'should', 'not', 'weren', 'because', 'll', 'by', 'will', 'so', 'same', 'isn', 'down', 've', 'they', "weren't", 'be', 'its', 'why', "you'd", 'above', 'yourselves', 'herself', "needn't", 'me', 'him', 'she']
    
    
    
  def sigmoid(self, value):
    '''
    Sigmoid function applies the sigmoid activation function on the given value.
    Parameters:
        value (int): value on which sigmoid activation is to be applied
        
    Return: 
        sig (float): sigmoid of the given value
    '''    
    sig = 1 / (1 + np.exp(-1*value))
    return sig

  def get_positive_words_count(self, counts):
    '''
    Returns the count of positive words in the given list or dict.
    Parameters:
        counts (list or dict): list or dict containing words
        
    Return: 
        pos_word_count (int): count of positive words in the list or dict based on ing Liu's sentiment lexicon
    '''        
    pos_word_count = 0
    for pos_word in self.pos_words:
      if pos_word in counts:
        pos_word_count += counts[pos_word]
    
    return pos_word_count

  def get_negative_words_count(self, counts):
    '''
    Returns the count of negative words in the given list or dict.
    Parameters:
        counts (list or dict): list or dict containing words
        
    Return: 
        pos_word_count (int): count of negative words in the list or dict based on ing Liu's sentiment lexicon
    '''            
    neg_word_count = 0
    for neg_word in self.neg_words:
      if neg_word in counts:
        neg_word_count += counts[neg_word]
    
    return neg_word_count

  def get_pronouns_count(self, counts):
    '''
    Returns the count of first and second person pronouns in the given list or dict.
    Parameters:
        counts (list or dict): list or dict containing words
        
    Return: 
        pronouns_count (int): count of first and second person pronouns in the list or dict
    '''            
    pronouns_count = 0
    for pronouns in self.pronouns:
      if pronouns in counts:
        pronouns_count += counts[pronouns]
    
    return pronouns_count

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """

    corpus = ""
    for sample in examples:
      idx, text, label = sample
      text = text.lower()      
      self.reviews.append(text)
      self.labels.append(label)
      corpus += text + ' '
    
    text = ' '.join(filter(lambda x: x not in self.stopwords, text.split()))

    # using NLTK Word Tokenizer for tokenization of train data
    corpus = word_tokenize(corpus)

    # word frequency maintains the count of all the vocabulary of the train corpus
    self.word_freq = Counter(corpus)

    # vocabulary of the train corpus
    self.vocab = list(self.word_freq.keys())

    # feature:  creating bag-of-words representation of all review for model training
    for review in self.reviews:
      counts = Counter(word_tokenize(review))
      vec = []
      for word in self.vocab:
        if word in counts:
          vec.append(counts[word])
        else:
          vec.append(0)

      # feature:  add positive word counts to vector
      pos_word_count = self.get_positive_words_count(counts)
      vec.append(pos_word_count)

      # feature:  add negative word counts to vector
      neg_word_count = self.get_negative_words_count(counts)
      vec.append(neg_word_count)

      # feature: add logarithm of review length to vector
      vec.append(np.log(len(review)))

      # feature: add count of pronouns to vector
      pronouns_count = self.get_pronouns_count(counts)
      vec.append(pronouns_count)

      # finally add the vector to the bow list
      self.bow.append(vec)

    # weights are initialized to 0 with length same as that of a single review vector length
    # 4 is added to account for the new features added
    self.weights = [0] * (len(self.vocab) + 4)
    
    # Gradient Descent Training
    for i in range(self.epochs):
      for idx, vec in enumerate(self.bow):
        product = np.dot(vec, self.weights) + self.bias
        output = self.sigmoid(product)
        gradient = (output - int(self.labels[idx])) * np.asarray(vec)
        self.weights = self.weights - (self.lr * gradient)

  def score(self, data):
    """
    Score a given piece of text
    you will compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """
    likelihoods = dict()
    data = data.lower()

    data = ' '.join(filter(lambda x: x not in self.stopwords, data.split()))

    counts = Counter(word_tokenize(data))
    vec = []
    for word in self.vocab:
      if word in counts:
        vec.append(counts[word])
      else:
        vec.append(0)

    # test sample is preprocessed the same way as training review vectors

    pos_word_count = self.get_positive_words_count(counts)
    vec.append(pos_word_count)

    neg_word_count = self.get_negative_words_count(counts)
    vec.append(neg_word_count)    

    vec.append(np.log(len(data)))

    pronouns_count = self.get_pronouns_count(counts)
    vec.append(pronouns_count)

    prob_positive = self.sigmoid(np.dot(self.weights, vec) + self.bias)
    prob_negative = 1 - prob_positive

    likelihoods[self.positive_label] = prob_positive
    likelihoods[self.negative_label] = prob_negative

    return likelihoods

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    scores = self.score(data)
    if (scores[self.positive_label] > scores[self.negative_label]) or (scores[self.positive_label] == scores[self.negative_label]):
      return self.positive_label
    else:
      return self.negative_label

  def featurize(self, data):
    """
    we use this format to make implementation of this class more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    features = []
    words = data.split()

    for word in words:
      if word in self.vocab:
        features.append((word, True))
      else:
        features.append((word, False))
    
    return features

  def __str__(self):
    # classifier name
    return "Logistic Regression Classifier (BOW, Pos/Neg word count, log review length, Pronouns count)"


