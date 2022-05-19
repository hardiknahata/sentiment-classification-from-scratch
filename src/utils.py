class EventsIO:

    def generate_tuples_from_file(self, training_file_path):
        """
        Generates tuples from file formated like:
        id\ttext\tlabel
        Parameters:
            training_file_path - str path to file to read in
        Return:
            a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        """
        f = open(training_file_path, "r", encoding="utf8")
        listOfExamples = []
        for review in f:
            if len(review.strip()) == 0:
                continue
            dataInReview = review.split("\t")
            for i in range(len(dataInReview)):
            # remove any extraneous whitespace
                dataInReview[i] = dataInReview[i].strip()
                t = tuple(dataInReview)
                listOfExamples.append(t)
        f.close()
        return listOfExamples

    def load_lexicon(self, filename):
        """
        Load a file from Bing Liu's sentiment lexicon
        (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), containing
        English words in Latin-1 encoding.
        
        Parameters:
            filename (str): path to the file
            
        Return: 
            lexicon(list): list of positive or negative lexicons
        """
        lexicon = []
        with open(filename, encoding='latin-1') as infile:
            for line in infile:
                line = line.rstrip()
                if line and not line.startswith(';'):
                    lexicon.append(line)
        return lexicon


class Metrics:
    def precision(self, gold_labels, predicted_labels):
        """
        Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
        Parameters:
            gold_labels (list): a list of labels assigned by hand ("truth")
            predicted_labels (list): a corresponding list of labels predicted by the system
        Returns: double precision (a number from 0 to 1)
        """
        correct_positive_preds = 0
        total_positive_preds = 0

        for i in range(len(predicted_labels)):
            if predicted_labels[i] == '1':
                total_positive_preds += 1

            if gold_labels[i] == '1':
                correct_positive_preds += 1
        
        return correct_positive_preds/total_positive_preds


    def recall(self, gold_labels, predicted_labels):
        """
        Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
        Parameters:
            gold_labels (list): a list of labels assigned by hand ("truth")
            predicted_labels (list): a corresponding list of labels predicted by the system
        Returns: double recall (a number from 0 to 1)
        """
        positive_preds = 0
        actual_positive_preds = 0

        for i in range(len(predicted_labels)):
            if gold_labels[i] == '1':
                actual_positive_preds += 1

            if predicted_labels[i] == '1':
                positive_preds += 1
        
        return positive_preds/actual_positive_preds


    def f1(self, gold_labels, predicted_labels):
        """
        Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
        Parameters:
            gold_labels (list): a list of labels assigned by hand ("truth")
            predicted_labels (list): a corresponding list of labels predicted by the system
        Returns: double f1 (a number from 0 to 1)
        """
        prec = self.precision(gold_labels, predicted_labels)
        rec = self.recall(gold_labels, predicted_labels)

        if prec == 0 and rec == 0:
            return 0

        f_score = (2*prec*rec) / (prec+rec)

        return f_score    